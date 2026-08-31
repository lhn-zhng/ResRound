# qkv/FC1 normal_threshold 分布与加速审计（r219）

## 约束与数据

- 保持 `normal_threshold` 选择结果和 ratio 不变。
- 数据来自 rank000 的 layer0/1、step1/500/1000，共六组 qkv 和六组 FC1。
- 形状分别为 qkv `8192x4096x3072`、FC1 `8192x4096x14336`。
- 全部短测期间后台长训练 PID 322641 保持运行。

## 列分布

| 模块 | actual ratio | 活跃行均值 | top-64 覆盖 | top-256 覆盖 | top-512 覆盖 | top-1024 覆盖 |
|---|---:|---:|---:|---:|---:|---:|
| qkv | `0.0909%--0.1032%` | `8056/8192` | `33.2%` | `53.4%` | `65.7%` | `78.8%` |
| FC1 | `0.0651%--0.1076%` | `7917/8192` | `55.0%` | `80.0%` | `89.3%` | `95.9%` |

FC1 的列偏斜很强，部分后期样本 top-64 已覆盖 `84.5%`；qkv 也有偏斜，
但 layer/step 波动明显更大。两者与 FC2 的关键差异是总 nnz 很小、但几乎所有行
仍活跃：每行中位数只有 `3--4` 个 outlier，却仍必须读写接近完整的 `M x N` 输出。

## hot/cold 结果

在 step1 layer0 上把 FC2 hot/cold 泛化到任意 `N % 16 == 0` 后：

| 模块 | 当前直接 correction | 最快 hot/cold | 结果 |
|---|---:|---:|---|
| qkv | `0.0685--0.0845 ms` | H=64，`0.1495 ms` | 更慢 |
| FC1 | vec16 `0.3385 ms` | H=64，`0.7747 ms` | 更慢 2.29x |

FC1 虽然 top-64 覆盖 `43.4%`，但 `8192x64x14336` 的 `addmm(beta=1)` 已需
`0.4117 ms`；cold kernel 又因几乎全行活跃维持约 `0.35 ms`。因此 separate
hot GEMM 无法利用列偏斜，不能接入默认路径。

## 4WG tile-local 结果

对同一 FC1 payload 复用旧 local-delta 4WG：

| 路径 | 时间 |
|---|---:|
| handwritten TMA dense | `0.8502 ms` |
| TMA dense + poststore | `1.2083 ms` |
| 4WG local delta | `1.5772 ms` |

local-delta 与 direct 的 `max_abs=0.03125`，且性能更慢。旧实现让 dense epilogue
等待 scalar sparse producer，不适合直接推广到 FC1。

## 当前可用优化

真实 weight transpose 成本：

| 模块 | 单层 BF16 transpose | 单层缓存 |
|---|---:|---:|
| qkv | `0.0956 ms` | `24 MiB` |
| FC1 | `0.4385 ms` | `112 MiB` |

启用缓存后，同一 TE weight workspace generation 内的 correction 短测从：

- qkv：`0.1646 -> 0.0863 ms`，降低 `47.6%`；
- FC1：`0.9448 -> 0.5128 ms`，降低 `45.7%`。

r219 最初仅依赖 storage identity 和 tensor version，这对 Megatron distributed
optimizer 不安全：它通过 `.data` 和共享 param buffer 更新权重，权重值变化时
`Parameter._version` 可以完全不变。r220 改为以 TE 首 microbatch 刷新的 weight
workspace generation 为主失效信号，并把缓存限制在对应 qresult 内；每次 workspace
更新还会显式清空旧 transpose。缺少 generation 时即使设置环境变量也强制旁路缓存。
CUDA Graph capture 时仍记录 transpose copy，避免 replay 使用旧权重。

缓存默认关闭。只应在短程 loss A/B 中实验性地缓存指定层：

```bash
export FP4_OUTLIER_FAST_FPROP_DIRECT_SPARSE_VARIANT=auto
export FP4_OUTLIER_FAST_FPROP_CACHE_WEIGHT_T=0
export FP4_OUTLIER_FAST_FPROP_CACHE_WEIGHT_T_LAYERS=linear_qkv,linear_fc1
```

24 层下约增加 qkv `0.56 GiB`、FC1 `2.63 GiB`。缓存只在同一 optimizer
step 的 weight workspace 被多个 microbatch/recompute 重用时节省时间；每步首次
FPROP 仍会刷新 transpose。在完成与无缓存长程 loss 对照前，不作为正式训练默认项。
正确性审计见 `docs/fp4_weight_t_cache_correctness_r220.md`。

## 下一版方向

FC1 真正值得做的是新 5WG deadline-bounded tile-local correction：

1. select/quant 同时输出按 M-tile/K 分组的小 payload；
2. 两个 sparse WG 提前一 tile 生成 local delta；
3. epilogue 只消费已 ready 的 bounded 条目，绝不等待 sparse；
4. 未按 deadline 完成的条目走 exact poststore fallback；
5. 不落地全局 compact delta，不再额外读写完整输出。

qkv 当前 correction 已只有约 `0.07 ms`，优先级低于 FC1；保留稳定 vstore 和
weight transpose 复用更合理。

## 结果文件

- `logs/r218_qkv_fc1_column_distribution_six_groups.json`
- `logs/r218_qkv_fc1_hotcold_step1_layer0_busy_gpu0.json`
- `logs/r218_qkv_fc1_direct_variants_six_groups_busy_gpu0.json`
- `logs/r219_qkv_fc1_weight_t_cache_integration_busy_gpu0.json`
