# FP4 select/quant 与 sparse correction 权衡（r212）

> FC2 部分已被 r216 推翻：保持 normal_threshold 原始 ratio 的 hot/cold correction 已显著快于 direct。当前结论见 `docs/fp4_fc2_hot_columns_r216.md`。本页其余 proj/qkv/fc1 结果保留为历史记录。

## 结论

- 不把完整 dense residual 构造放进 select/quant。它会重复写 `heavy_rows x K`，历史 direct-split 实测增加约 62--116 us，完整排序 schedule 在真实 proj 上把 select 从 0.1842 ms 拉到 0.6173 ms。
- select 内只复用已经生成的 `row_counts`，用 block-level compact 生成 padded active rows；未使用槽填 `-1`，下游 kernel 直接跳过，不读取 host count。
- heavy/light 分组留在 sparse correction 侧，用单个异步 CUDA kernel 生成 padded light 列表和固定容量 heavy 列表。heavy 超容量行自动回落 direct，不丢 correction。
- `auto` 利用持久 quantizer 的 `layer_name`：首样本同步一次得到 actual ratio，按层缓存策略；后续恢复 deferred、无 host sync。缓存按 `(M, K)` 隔离。

## 真实 linear_proj 结果

数据：rank000 step1 layer0，`M=16384, K=2048, N=4096`，配置 ratio 0.1%，实际 ratio 0.560585%，188101 个 outlier。GPU 忙载下中位数。

| 路径 | select+quant (ms) | sparse (ms) | dense+sparse qgemm (ms) | fresh-qx cached-weight (ms) |
|---|---:|---:|---:|---:|
| 原 deferred，无 schedule | 0.1842 | 0.4901 | 0.6370 | 1.4058 |
| CUB sorted schedule | 0.6173 | 0.5125 | 0.6654 | 1.3460 |
| unsorted + host count | 0.5561 | 0.5003 | - | 1.2418 |
| padded active，完全 deferred | 0.1908 | 0.5126 | - | 0.8539 |
| padded light/heavy，阈值64/容量512 | 0.1844 | 0.2911 | 0.4488 | **0.6351** |

同次 BF16 GEMM 为 0.6969 ms；最终 fresh-qx 路径快 8.9%。相对原 fresh 串行 1.4058 ms，降低 54.8%。

阈值 64 时共有 7956 light rows、507 heavy rows；507 heavy rows承载 125336/188101（66.6%）outliers。新旧 hybrid 在容量 512 时输出逐 bit 一致。

## 跨模块边界

| 模块 | 形状 | 实际 ratio | 推荐 | fresh-qx (ms) | BF16 (ms) |
|---|---|---:|---|---:|---:|
| linear_qkv | `8192x4096x3072` | 0.1004% | direct vstore；weight-T cache 默认关闭 | 0.4976 | 0.6024 |
| linear_fc1 | `8192x4096x14336` | 0.0908% | direct vstore；weight-T cache 默认关闭 | 1.7099 | 4.7801 |
| linear_proj 重尾层 | `16384x2048x4096` | 0.5606% | padded hybrid 64/512 | 0.6351 | 0.6969 |
| linear_fc2 | `16384x7168x4096` | 1.9308% | r216 hot/cold，ratio 不变 | correction 1.04--1.69 ms | 见 r216 配对结果 |

旧 direct/hybrid 在 FC2 上确实超过经济区，但 r216 利用 normal_threshold 的列频偏斜，把高频 K 列改成小型 dense GEMM、剩余条目保留 exact sparse correction。无需降低 ratio，也无需修改 proj。

## 多 step 稳定性

同形状 proj 的 heavy64 行数在 step/layer 间为 `507, 14, 329, 0, 348, 0`。因此不能只按 shape 强制 hybrid。`auto` 仅在 layer name 包含 `linear_proj`、形状为 `16384x2048`、首样本 actual ratio >= 0.3% 时缓存 hybrid，其余缓存 direct。

固定 heavy 容量溢出时，超出行回落 direct。故意使用容量 128 时，379 行正确回落，light 数从 7956 增至 8335；与全 heavy BF16 GEMM 的最大差为 0.001953125，来源是 FP32-direct 与 BF16-GEMM 的舍入路径不同，不是数据丢失。

## 推荐实验环境

```bash
export FP4_OUTLIER_FAST_FPROP_TRUST_CAPACITY=1
export FP4_OUTLIER_FAST_FPROP_ASSUME_NO_OVERFLOW=1
export FP4_OUTLIER_FAST_FPROP_DEFER_SELECTED_SYNC=1
export FP4_OUTLIER_FAST_FPROP_DIRECT_NOMASK=1
export FP4_OUTLIER_FAST_FPROP_COLUMNWISE_SOURCE=direct
export FP4_OUTLIER_FAST_FPROP_SPARSE_POLICY=auto
export FP4_OUTLIER_FAST_FPROP_SPARSE_AUTO_CACHED=1
export FP4_OUTLIER_FAST_FPROP_SPARSE_PADDED_HYBRID_MIN_ACTUAL_RATIO=0.003
export FP4_OUTLIER_FAST_FPROP_SPARSE_HEAVY_THRESHOLD=64
export FP4_OUTLIER_FAST_FPROP_SPARSE_PADDED_HEAVY_CAPACITY=512
export FP4_OUTLIER_FAST_FPROP_DIRECT_SPARSE_VARIANT=auto
# 仅用于 r220 短程 loss A/B；正式长训练保持关闭：
# export FP4_OUTLIER_FAST_FPROP_CACHE_WEIGHT_T_LAYERS=linear_qkv,linear_fc1
```

`auto` 首样本会缓存 1.5 倍 selected 的后续 capacity hint。`TRUST_CAPACITY/ASSUME_NO_OVERFLOW` 仍应只在已验证 shape 上使用；首次超过容量会回退，后续容量按 hint 扩大。分布可能突变的训练应保留更大 shape-specific headroom。

weight-T cache 的 r219 tensor-version 失效策略不适用于 distributed optimizer 的
`.data`/共享 buffer 更新。r220 已改为 TE weight workspace generation，但在完成长程
loss A/B 前仍保持默认关闭；详见 `docs/fp4_weight_t_cache_correctness_r220.md`。

## 结果文件

- `logs/r212_actual_proj_padded_active_deferred_busy_gpu1.json`
- `logs/r212_actual_proj_dense_light_padded64_c512_busy_gpu1.json`
- `logs/r212_actual_proj_dense_light64_deferred_busy_gpu1.json`
- `logs/r212_matrix_direct_linear_qkv_busy_gpu1.json`
- `logs/r212_matrix_direct_linear_fc1_busy_gpu1.json`
- `logs/r212_matrix_direct_linear_fc2_busy_gpu1.json`
- `logs/r212_matrix_padded_linear_qkv_busy_gpu1.json`
- `logs/r212_matrix_padded_linear_fc1_busy_gpu1.json`
- `logs/r212_matrix_padded_linear_fc2_busy_gpu1.json`
