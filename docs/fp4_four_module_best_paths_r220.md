# 四模块当前最佳 FPROP 路径与基线对比（r220）

## 口径

- 保持各层原始 `normal_threshold` 结果，不降低 ratio，不修改 proj 选择策略。
- `best full` 包含 input select/quant、NVFP4 dense GEMM 和 exact sparse correction。
- `native dense` 为同形状 TE input quant + NVFP4 dense GEMM，weight qresult 已按 TE
  标准 microbatch workspace 复用；不包含 outlier select/correction。
- BF16 列为同轮 BF16 GEMM。
- 自定义 BF16 weight-T cache 关闭；表中“cached weight”仅指 TE 原生量化 weight
  workspace，不是 r220 实验性 transpose cache。
- qkv/FC1/proj 为 step1 layer0 实测；FC2 `3.46 ms` 是后台训练存在时的干净组件
  推算，不是无竞争端到端实测。

## 汇总

| 模块 | 当前最佳安全路径 | best full | native dense | 相比 native dense | BF16 | 相比 BF16 | 相比旧串行 |
|---|---|---:|---:|---:|---:|---:|---:|
| qkv | direct vstore poststore | `0.4976 ms` | `0.3281 ms` | 慢 `51.7%` | `0.6024 ms` | 快 `17.4%` | 算法仍是串行；相对 fully-fresh `0.5549 ms` 快 `10.3%` |
| FC1 | direct vstore poststore | `1.7099 ms` | `0.8575 ms` | 慢 `99.4%` | `4.7801 ms` | 快 `64.2%` | 算法仍是串行；相对 fully-fresh `1.9048 ms` 快 `10.2%` |
| proj 重尾层 | padded light/heavy `64/512` | `0.6351 ms` | `0.3706 ms` | 慢 `71.4%` | `0.6969 ms` | 快 `8.9%` | `1.4058 -> 0.6351 ms`，快 `54.8%` |
| FC2 | adaptive hot/cold，H=`1024--2048` | 约 `3.46 ms` | `1.6038 ms` | 慢约 `115.7%` | 约 `4.70 ms` | 快约 `26.4%` | 配对忙载快 `42.8%`；干净组件约快 `45--49%` |

qkv/FC1 的 fully-fresh 对照每次都重新量化 weight；训练中 TE 会在同一 optimizer
step 的后续 microbatch 复用 weight workspace，所以 `best full` 更接近稳态。这个约
`10%` 的差异不是 sparse/dense overlap。

## Sparse 是否隐藏

只看已量化输入后的 dense+sparse qgemm critical path：

| 模块 | dense GEMM | dense+sparse | 仍可见开销 |
|---|---:|---:|---:|
| qkv | `0.1744 ms` | `0.3179 ms` | `+82.3%` |
| FC1 | `0.7183 ms` | `1.5094 ms` | `+110.1%` |
| proj 重尾层 | `0.2293 ms` | `0.4488 ms` | `+95.7%` |
| FC2，无 weight-T cache 组件估计 | `0.7126 ms` | 约 `2.3944 ms` | 约 `+236%` |

因此四个模块当前都没有把 sparse correction 完全隐藏在 dense GEMM 内。proj/FC2 的
收益来自把昂贵 direct sparse 串行路径改写成更适合其分布的 correction，而不是让
correction 变成零成本。

## 模块策略

1. qkv：保持 direct vstore。ratio 很低，correction kernel 约 `0.07 ms`；hot/cold
   和旧 4WG 的固定成本不划算。
2. FC1：保持 direct vstore。列偏斜虽强，但几乎全行活跃，separate hot GEMM 更慢；
   下一步应做 deadline-bounded tile-local 5WG，而不是现有 compact-delta 4WG/5WG。
3. proj：按首样本 actual ratio/重尾程度自适应。重尾层使用 padded light/heavy
   `64/512`，低重尾层回退 direct，不能按 shape 全部强制 hybrid。
4. FC2：使用 adaptive hot/cold，高频 K 列走小 BF16 GEMM，cold 使用 vec16 exact
   correction 和容量不足回退；不要启用旧双流 overlap 或当前 compact-delta 4WG/5WG。

## 数据来源

- `logs/r212_matrix_direct_linear_qkv_busy_gpu1.json`
- `logs/r212_matrix_direct_linear_fc1_busy_gpu1.json`
- `logs/r212_actual_proj_dense_light_padded64_c512_busy_gpu1.json`
- `logs/r216_fc2_direct_paired_original_ratio_w2i5_busy_gpu0.json`
- `logs/r216_fc2_hotcold_h1536_original_ratio_w2i5_busy_gpu0.json`
- `docs/fp4_fc2_hot_columns_r216.md`
