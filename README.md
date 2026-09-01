# ResRound

Reference implementation for **ResRound: Improving Native NVFP4 Training via
Sparse Activation Residuals and Activation-Aware Weight Rounding**.

**Authors:** [LuHan Zhang](https://openreview.net/profile?id=~LuHan_Zhang1),
[Xinrui Yang](https://openreview.net/profile?id=~Xinrui_Yang5), and
[Shaohuai Shi](https://openreview.net/profile?id=~Shaohuai_Shi1) (corresponding
author).

## Overview

ResRound improves native NVFP4 training with two forward-pass mechanisms:

- **Sparse activation residuals (SAR):** preserve selected activation
  residuals and apply a sparse BF16 correction alongside dense NVFP4
  computation.
- **Activation-aware weight rounding (AWR):** adapt weight rounding using
  activation statistics.

This source-only research release contains the ResRound integration for
Megatron-LM, the fused activation selection and quantization kernels, the dense
NVFP4 plus sparse BF16 execution path, and the experiment entry points used in
the paper.

## Code map

- `megatron/core/extensions/fp4_outlier/`: SAR selection, quantization,
  storage, execution, and AWR.
- `profile/r203_dense_sparse_fusion_report_20260629/kernels/rowcol_quant_packed_r207/`:
  fused activation selection and NVFP4 quantization extension.
- `collected/nvfp4_warpgroup_sparse_fusion/`: CUDA sources and Python loader
  for dense NVFP4 computation with sparse BF16 correction.
- `examples/fp4_paper_repro/`: scripts for the paper experiments and component
  comparisons.

See [RESROUND_RELEASE.md](RESROUND_RELEASE.md) for the tested environment,
dependencies, setup details, and reproduction commands.

## Quick start

Install this repository and a compatible Transformer Engine build in the same
environment:

```bash
export RESROUND_ROOT=/path/to/ResRound
export TRANSFORMER_ENGINE_ROOT=/path/to/TransformerEngine
export FP4_OUTLIER_FAST_KERNEL_ROOT="$RESROUND_ROOT"
export PYTHONPATH="$RESROUND_ROOT:${PYTHONPATH:-}"
cd "$RESROUND_ROOT"
pip install -e .
```

The CUDA extensions are compiled lazily on first use. On Blackwell B200-class
systems, set `TORCH_CUDA_ARCH_LIST=12.0a` if PyTorch does not select the
architecture automatically.

Example reproduction command:

```bash
DATA_PATH=/path/to/indexed/dataset \
TOKENIZER_MODEL=/path/to/tokenizer \
VARIANT=local_cov_r2_s1024 \
bash examples/fp4_paper_repro/train_quartet_c4_100m_local_cov_compare.sh
```

## Release scope

Datasets, tokenizers, checkpoints, experiment logs, W&B state, compiled
extensions, and benchmark caches are intentionally excluded. Supply local
paths through the environment variables documented in the reproduction
scripts. Set `WANDB_MODE=disabled` for a fully local run.

## Acknowledgements and license

This implementation is built on
[NVIDIA Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and retains its
Apache-2.0 license and copyright notices. See [LICENSE](LICENSE).
