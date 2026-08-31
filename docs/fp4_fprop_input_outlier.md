# FP4 FPROP Input Outlier Recipe

This branch keeps only the FPROP input-outlier path:

- `linear_input`: split `A = A1 + A2`; `A1` is quantized with TE NVFP4 and `A2` is kept as sparse exact correction.
- `linear_weight`: uses TE NVFP4 for the dense main GEMM and keeps a BF16 dense reference for `A2 @ W`.
- `linear_grad_output`: uses the default TE NVFP4 path. Custom DGRAD/WGRAD outlier paths are intentionally not present.
- Optional TP+SP FPROP input all-gather: local ranks split first, then TE gathers `NVFP4(A1)` and the custom path gathers the sparse `A2` payload.

Direct CLI usage:

```bash
--fp4-format e2m1 \
--fp4-recipe custom \
--fp4-quantizer-factory megatron.core.extensions.fp4_outlier_recipe.nvfp4_outlier_quantizer_factory \
--fp4-outlier-ratio 0.01 \
--fp4-outlier-selection-method normal_threshold \
--fp4-outlier-enable-fast-fprop
```

Precision-config usage:

```bash
--te-precision-config-file megatron/core/extensions/fp4_outlier_config.yaml
```

Supported knobs:

- `--fp4-outlier-ratio`
- `--fp4-outlier-selection-method {topk,normal_threshold}`
- `--fp4-outlier-adaptive-ratio`
- `--fp4-outlier-adaptive-min-ratio`
- `--fp4-outlier-adaptive-max-ratio`
- `--fp4-outlier-adaptive-reference-heaviness`
- `--no-fp4-outlier-enable-fprop`
- `--fp4-outlier-enable-fast-fprop`
- `--fp4-outlier-store-input-dense-main`
- `--fp4-outlier-main-quantizer-rht`
- `--fp4-outlier-input-stochastic-rounding`
- `--fp4-outlier-enable-nvfp4-a1-a2-all-gather`
- `--no-fp4-outlier-enable-dgrad`
- `--no-fp4-outlier-enable-wgrad`

`fp4_outlier_enable_dgrad` and `fp4_outlier_enable_wgrad` must remain `False` in this branch.

`fp4_outlier_enable_fast_fprop` keeps the same FPROP decomposition but swaps in the optional
fast kernels when they are compatible:

- `linear_input` select+quant: r207 packed row/column NVFP4 select+quant for
  `normal_threshold`.
- FPROP sparse correction: direct BF16 sparse post-store correction into the dense-main GEMM
  output instead of `torch.sparse.mm`.

The fast select+quant path intentionally does not replace exact `topk`; when
`fp4_outlier_selection_method=topk`, or when the external kernels/shape/dtype are unsupported, the
recipe falls back to the original implementation and logs a rank-0 fallback message once.

The llama example script can run both sides of the throughput comparison with the same model/data
settings:

```bash
# TE native NVFP4 baseline.
FP4_RECIPE=nvfp4 examples/llama/train_llama3_8b_h100_fp4_fprop_input_outlier_r003_ag.sh

# Custom FPROP input-outlier path with fast select+quant and sparse correction.
FP4_RECIPE=custom FP4_OUTLIER_ENABLE_FAST_FPROP=1 \
  examples/llama/train_llama3_8b_h100_fp4_fprop_input_outlier_r003_ag.sh
```

`fp4_outlier_enable_nvfp4_a1_a2_all_gather` is intended for tensor-parallel sequence-parallel
ColumnParallel FPROP input gather. It keeps the split local to each rank, communicates TE-native
NVFP4 storage for `A1`, and communicates sparse `A2` separately. This is not bitwise equivalent to
gathering the BF16 input first and then selecting outliers globally.

When `fp4_outlier_adaptive_ratio` is enabled, the configured ratio is converted per tensor to:

```text
effective_ratio = fp4_outlier_ratio * log(max(abs(A)) / mean(abs(A))) / log(reference_heaviness)
```

The result is clamped to `[adaptive_min_ratio, adaptive_max_ratio]`, with values below `1e-5`
treated as zero. The resolved effective ratio is then used by both `topk` and `normal_threshold`
selection.
