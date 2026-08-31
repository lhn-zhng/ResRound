# Weight-T cache 训练正确性审计（r220）

## 结论

r219 仅用 `Tensor._version` 判断 BF16 weight transpose 是否失效，不足以保证
Megatron 训练正确性。distributed optimizer 会把更新后的 main parameter 写入共享
`param_data` buffer，常见路径包含 `.data.copy_`、buffer slice copy 和 collective
写入；这些写入可以改变参数值而不改变 `Parameter._version`。这能解释旧缓存打开后
loss 崩溃的现象。

缓存继续默认关闭。r220 增加 TE weight workspace generation 门禁后才允许进入缓存，
但在完成配对长程 loss 实验前仍只作为实验路径。

## 最小复现

在 `transformer_engine` conda 环境中：

| 更新方式 | 参数值变化 | `Parameter._version` |
|---|---:|---:|
| `param.data.copy_(...)` | 是 | `0 -> 0` |
| 共享 buffer `copy_` | 是 | 不变 |
| 共享 buffer `add_` | 是 | 不变 |
| `no_grad(): param.copy_(...)` | 是 | `0 -> 1` |

因此 storage pointer、offset、shape、stride、`_version` 全部相同并不代表权重未更新。

## r220 失效协议

1. 每次 custom `linear_weight` quantizer 构造或刷新 TE weight workspace 时，递增
   `weight_update_generation`。
2. Megatron 在每个 optimizer step 后的首 microbatch 设置
   `is_first_microbatch=True`，TE 因而调用 workspace `quantize_` 刷新该 generation。
3. transpose cache 只挂在对应 weight qresult 上，不再从持久 quantizer owner 跨
   workspace 复用。
4. generation 改变时复用 buffer 但重新执行 `weight.t()` copy；quantizer workspace
   update 同时显式清空旧 cache state。
5. qresult 缺少 generation 时，即使环境变量请求缓存也强制旁路。
6. CUDA Graph capture 继续记录 transpose copy，优先保证 replay 不读取旧权重。

## 定向验证

使用 `normal_threshold`、128x128 BF16 correction：

- 通过 `weight.data.add_(0.25)` 模拟不会 bump version 的优化器更新；
- `_version` 保持 `0 -> 0`；
- workspace generation 刷新为 `1 -> 2`；
- 刷新后的 cached correction 与禁用缓存 reference：`max_abs=0`；
- 刷新后结果与旧权重 correction：`max_abs=2.875`，确认测试确实观察到了权重更新。
- 连续 8 次 `.data` 更新的 generation 为 `1..8`，每步两个 microbatch、共 16 次
  cached-vs-reference correction 的最大误差仍为 `0`。
- 通过 TE `get_weight_workspace(update_workspace=True)` 的真实刷新入口验证：workspace
  对象保持复用、generation 递增，旧 transpose state 被清空。

当前后台长训练 PID 322641 显式设置
`FP4_OUTLIER_FAST_FPROP_CACHE_WEIGHT_T=0`，未使用该缓存路径。

## 启用边界

正式训练继续使用：

```bash
export FP4_OUTLIER_FAST_FPROP_CACHE_WEIGHT_T=0
unset FP4_OUTLIER_FAST_FPROP_CACHE_WEIGHT_T_LAYERS
```

后续应先做固定 seed、相同 checkpoint 的 cache-off/cache-on 配对短训，逐 step 比较
loss、grad norm、参数 checksum 和 skipped iteration；通过后再扩大训练长度。
