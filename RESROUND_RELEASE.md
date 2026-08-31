# ResRound release notes

## Scope

This is a source-only research release. It intentionally excludes datasets,
checkpoints, experiment logs, W&B state, compiled extensions, and benchmark
cache files. Dataset, tokenizer, checkpoint, and output locations are supplied
through environment variables in the scripts under
`examples/fp4_paper_repro/`.

The implementation is based on NVIDIA Megatron-LM commit
`d9978209124e93a1fdb52e26e7a042fd166a1615`. It contains the ResRound
integration stack and the final local working-tree changes used for the paper
experiments.

## Code map

- `megatron/core/extensions/fp4_outlier/`: SAR selection, quantization,
  storage, execution, and AWR (`weight_rounding.py`).
- `profile/r203_dense_sparse_fusion_report_20260629/kernels/`
  `rowcol_quant_packed_r207/`: fused activation selection and NVFP4
  quantization extension.
- `collected/nvfp4_warpgroup_sparse_fusion/`: CUDA sources and Python loader
  for the dense NVFP4 plus sparse BF16 correction path.
- `examples/fp4_paper_repro/train_gpt2_345m_openwebtext.sh`: GPT-2 345M
  BF16, TE-NVFP4, SAR, AWR, and ResRound variants.
- `examples/fp4_paper_repro/train_llama2_1p24b_*`: Llama 1.24B training
  variants used by the long-horizon study.
- `examples/fp4_paper_repro/train_quartet_c4_100m_local_cov_compare.sh`:
  compact component comparison.

Some internal source symbols retain the development name `ELCR`; in the paper
and this release, that mechanism is called activation-aware weight rounding
(AWR).

## Tested environment

The paper runs were produced on NVIDIA Blackwell GPUs with:

- Python 3.12
- PyTorch 2.8.0 + CUDA 12.8
- CUDA toolkit 12.8
- Transformer Engine 2.12 development build at commit `af5cb431`

The matching public Transformer Engine source is available from the
`reduce_scatter_mixed` branch of
`https://github.com/ffhh927/TransformerEngine`.

## Setup

Install this repository and a compatible Transformer Engine build in the same
environment. Point the runtime at both source trees:

```bash
export RESROUND_ROOT=/path/to/ResRound
export TRANSFORMER_ENGINE_ROOT=/path/to/TransformerEngine
export FP4_OUTLIER_FAST_KERNEL_ROOT="$RESROUND_ROOT"
export PYTHONPATH="$RESROUND_ROOT:${PYTHONPATH:-}"
cd "$RESROUND_ROOT"
pip install -e .
```

The two CUDA extensions are compiled lazily on first use. Set
`TORCH_CUDA_ARCH_LIST=12.0a` for Blackwell B200-class systems if PyTorch does
not select the architecture automatically.

## Reproduction entry points

All paths below can be overridden without editing the scripts. At minimum,
set the dataset and tokenizer paths required by the chosen model.

```bash
# 100M component comparison
DATA_PATH=/path/to/indexed/dataset \
TOKENIZER_MODEL=/path/to/tokenizer \
VARIANT=local_cov_r2_s1024 \
bash examples/fp4_paper_repro/train_quartet_c4_100m_local_cov_compare.sh

# GPT-2 345M long-horizon variants
DATA_PATH=/path/to/openwebtext_prefix \
RUN_VARIANTS=bf16,te,our,weight_only,weight_input_r0p003 \
bash examples/fp4_paper_repro/train_gpt2_345m_openwebtext.sh

# Llama 1.24B ResRound variant
DATA_PATH=/path/to/refinedweb_prefix \
TOKENIZER_MODEL=/path/to/llama_tokenizer \
bash examples/fp4_paper_repro/train_llama2_1p24b_joint_residual_qkv8_proj22_from0_4gpu.sh
```

Set `WANDB_MODE=disabled` for a fully local run. Training scripts default to
writing logs, checkpoints, TensorBoard data, and dataset caches under the
repository; these directories are ignored by Git.

## License and upstream

This repository retains the upstream Megatron-LM Apache-2.0 license and its
copyright notices. Before making the repository public, the authors should
confirm the license that applies to newly added ResRound-specific files.
