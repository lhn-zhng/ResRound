# FC2 normal_threshold hot/cold correction（r216）

## 约束

- 保持 `normal_threshold` 与配置 ratio 不变。
- step1 layer0 仍选择 `2,267,535` 个 outlier，actual ratio 仍为 `1.9307945%`。
- 不修改 `linear_proj` 的 ratio 或策略。
- cold padded 容量不足时，自动回读原始 row payload；不截断、不丢 correction。

## 根因与方案

FC2 形状为 `M=16384, K=7168, N=4096`。原 row-major correction 对每个 outlier 重读一整行权重，真实 payload 又具有明显列偏斜：

- 全部 `2,267,535` 个 outlier 分布在 6,971 个 K 列；
- 最热 K 列出现 10,564 次；
- 前 1,536 个 K 列覆盖 65.8%--98.2% 的 outlier，取决于 layer/step；
- 128-row K-major 理论可减少 80.49% 的 B 向量读取，但旧 atomic 核与新 serial-shared 累加核都远慢于 row-major。

r216 因此把同一份 A2 payload 仅重排计算：

1. 高频 K 列写入小型 BF16 `hot_dense`，用 `addmm(beta=1)` 计算。
2. 剩余条目按行写入固定 stride 的 padded cold payload，使用 vec16 CUDA 核修正。
3. 每 64 次按当前列频刷新 hot 列。
4. 默认从 H=1024 开始；每增加 256 列若可再吸收至少 85,000 个 cold entries，则继续增加，最大 H=2048。
5. 默认 cold capacity 为 256；超容量行在设备端走原始 payload 精确回退。

临时张量不挂在 qx 上，因此不会在所有层 forward 后累计。H=2048 时主要 scratch 约为 96 MiB，并由 CUDA allocator 按流复用。

## 配对结果

数据为 rank000 step1 layer0，原始 `normal_threshold` ratio；GPU0 同时有长训练，因此跨多个 kernel 的完整事件会包含外部 CUDA context 切片。下表 direct 与 hot/cold 在相邻运行中使用相同负载、warmup/iters=`2/5`。

| 路径 | sparse correction | dense+sparse qgemm | fresh-qx cached-weight |
|---|---:|---:|---:|
| direct row-major | 4.9312 ms | 5.5076 ms | 13.5649 ms |
| hot/cold，H=1536 | **1.4335 ms** | **2.0608 ms** | **7.7644 ms** |
| 改善 | **3.44x** | **-62.6%** | **-42.8%** |

最终输出对 BF16 reference：

- direct：`max_abs=0.0693359375`，`rel_l2=0.1047837883`；
- hot/cold：`max_abs=0.0693359375`，`rel_l2=0.1047948897`。

hot/cold 多一次 BF16 汇合，造成很小的舍入差异；没有 outlier 丢失。强制 `cold_capacity=1` 时仍得到 `rel_l2=0.104795`，验证了超容量精确回退。

## 六个真实样本

layer0/1、step1/500/1000，actual ratio 为 1.841%--1.946%。自适应 H 的 correction 结果：

- 平均：`1.2063 ms`；
- 最快：`0.8869 ms`；
- 最慢：`1.6853 ms`；
- dense qgemm 与 sparse correction 的组件和平均：`1.9156 ms`；
- 同组 BF16 GEMM 平均：`4.7448 ms`。

后台训练使该轮 r207 select+quant 事件被放大到约 5.7 ms；历史同 payload、无该次 context 切片的稳定值约为 1.063 ms。因此 `1.063 + 1.916 ≈ 2.98 ms` 只能视为干净 GPU 上的组件推算，最终长训练吞吐仍需在无第二训练进程竞争时确认。不开 BF16 weight transpose cache时，step1 layer0 的 correction 为 1.682 ms，对应组件推算约 3.46 ms，仍低于同轮 BF16 4.70 ms。

## 被否决的方向

| 方向 | 结果 | 结论 |
|---|---:|---|
| cuSPARSE CSR `addmm` | 5.995 ms | 比 direct 5.018 ms 慢 |
| 全部 A2 densify + BF16 GEMM | sparse 5.610 ms | 计算量过大 |
| K-major shared atomic | 59.48 ms | shared atomic 主导 |
| K-major serial shared accumulator | 46.5--132.6 ms | shared RMW 与串行 group 主导 |
| hot/cold 双流 overlap | 1.676 ms vs serial 1.357 ms | 资源竞争与 merge 使其慢 23.5% |
| cold metadata shared staging | 约 0.5% | 小于共享 GPU 噪声，不保留 |

### cold 与主 GEMM 的 4WG/5WG 同核重叠验证

对 step1 layer0 的自适应 `H=1792` payload，cold 为 `414,001` 个条目，
占输入 `0.352520%`；但 `16,383/16,384` 行仍然活跃。因此 production
4WG/5WG 必须生成近似完整的 `M x N` compact delta，随后再做一次全输出 merge。

在后台长训练保持运行时，以同一份真实 cold CSR 做短测：

| 路径 | 时间 |
|---|---:|
| TE 主 NVFP4 GEMM | `0.7106 ms` |
| 当前 cold vec16 correction | `0.4612 ms` |
| handwritten TMA dense | `0.8288 ms` |
| 4WG cold producer | `2.8040 ms` |
| 4WG producer + merge | `3.1181 ms` |
| 5WG cold producer | `1.6878 ms` |
| 5WG producer + merge | `2.0019 ms` |
| compact merge 单独 | `0.3259 ms` |

即使假设 cold 计算能被完全免费隐藏，`TMA dense + merge = 1.1547 ms`，
相对当前 `TE dense + cold = 1.1718 ms` 的理论上限也只有约 `0.017 ms`；
实际 5WG complete 则慢约 `0.830 ms`。4WG/5WG 输出和当前 exact cold reference
的 `max_abs=0.001953125`，因此失败原因是性能而不是正确性。

结论：不能直接把当前 production 4WG/5WG compact-delta 路径用于 FC2 cold。
只有在消掉完整 delta merge、并让 bounded tile-local cold 在 dense epilogue 前完成时，
同核重叠才可能有收益；现有 pre-store/local-delta 历史实验已表明这会把 sparse
重新放到 dense critical path，不能作为当前默认方案。

## 默认与可选参数

默认 `SPARSE_POLICY=auto` 会且只会对满足以下条件的层选择该路径：

- layer name 包含 `linear_fc2`；
- 输入形状为 `16384x7168`；
- selection method 为 `normal_threshold`；
- actual ratio 至少 0.5%。

可选参数：

```bash
export FP4_OUTLIER_FAST_FPROP_FC2_HOT_REFRESH_INTERVAL=64
export FP4_OUTLIER_FAST_FPROP_FC2_COLD_CAPACITY=256
# 固定 H 时才设置；不设置则使用自适应 1024--2048。
# export FP4_OUTLIER_FAST_FPROP_FC2_HOT_COLS=1536
# 速度优先、可接受每层缓存 BF16 weight transpose 时设置。
# export FP4_OUTLIER_FAST_FPROP_CACHE_WEIGHT_T=1
```

不要为该路径设置 `FP4_OUTLIER_FAST_FPROP_THRESHOLD_SIGMA_BY_SHAPE`；r216 的目标就是保持 normal_threshold 算出的原始 ratio。

## 结果文件

- `logs/r216_fc2_direct_paired_original_ratio_w2i5_busy_gpu0.json`
- `logs/r216_fc2_hotcold_h1536_original_ratio_w2i5_busy_gpu0.json`
- `logs/r216_fc2_hotcold_h1536_six_groups_original_ratio_w1i3_busy_gpu0.json`
- `logs/r216_fc2_hotcold_auto_six_groups_original_ratio_w1i2_busy_gpu0.json`
- `logs/r216_fc2_hotcold_cap1_exact_fallback_w1i2_busy_gpu0.json`
- `logs/r216_fc2_final_default_auto_original_ratio_w1i2_busy_gpu0.json`

全部编译和短测期间，后台长训练 PID 322641 始终存活，未发送信号、未重启进程。
