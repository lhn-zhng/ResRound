# AAAI Experiment Readiness Checklist

Last updated: 2026-07-09.

## Target Scope

- Main convergence experiment aligned with AGoQ arXiv:2605.00539:
  - LLaMA2-7B
  - 2B training tokens
  - OpenWebText-style data
  - downstream zero-shot tasks: ARC-Challenge, ARC-Easy, HellaSwag, PIQA, SciQ, WinoGrande
- Small-model convergence checks:
  - current 100M Llama-like 6k run
  - one additional mid-size model still needs to be selected, for example 560M or 1.7B/1.9B
- System/throughput experiments:
  - can use larger or more communication-heavy model shapes than the convergence runs
  - should include BF16, TE NVFP4, ours 0.1%, and Quartet-II last

## Already Available

### Implementation

- Repo for the paper method path:
  - `/workspace/Megatron-LM-312-fprop-input`
- Current fast FPROP setting to preserve:
  - `FP4_OUTLIER_ENABLE_FAST_FPROP=1`
  - `FP4_OUTLIER_FAST_FPROP_DIRECT_SPARSE_VARIANT=sum_then_add`
  - `FP4_OUTLIER_FAST_FPROP_CACHE_WEIGHT_T=0`
  - `NVFP4_DIRECT_ADD_TMA_VARIANT=sumthenadd`
  - `FP4_OUTLIER_SPARSE_CORRECTION_BACKEND=auto`
- Correctness checks already run:
  - component fast-vs-reference check passed for qkv/proj/fc1/fc2 shapes
  - 250-step 100M convergence diagnostic passed with no skipped/nan
- Active 6k small-model run:
  - script: `examples/fp4_paper_repro/train_quartet_c4_100m_fprop_input_6k_ours0p1.sh`
  - W&B: `https://wandb.ai/1348564418-hit/megatron_fp4_paper_repro_train_quartet_c4_100m/runs/hgs76bxj`

### Existing Scripts To Reuse Or Port

- In `/workspace/Megatron-LM-312-fprop-input`:
  - `examples/fp4_paper_repro/train_quartet_c4_100m_fprop_input_6k_ours0p1.sh`
  - `examples/llama/train_llama3_8b_h100_fp4_fprop_input_outlier_r003_ag.sh`
  - component and timing tools under `tools/benchmark_fp4_*` and `tools/debug_fp4_fprop_fast_vs_ref.py`
- In `/workspace/Megatron-LM-312`:
  - `examples/fp4_paper_matrix/common_gpt_fp4_matrix.sh`
  - `examples/fp4_paper_matrix/train_llama_like_100m.sh`
  - `examples/fp4_paper_matrix/train_nanochat_like_560m.sh`
  - `examples/fp4_paper_matrix/train_nanochat_like_1p9b.sh`
  - `examples/fp4_paper_matrix/train_qwen3_1p7b.sh`
  - `examples/fp4_paper_repro/RUNBOOK.md`
  - `examples/fp4_paper_repro/DATASETS.md`

### Data And Models

- Pretraining indexed dataset prefixes available:
  - `/share/datasets/enwiki_openwebtext_llama3/merge`
  - `/share/datasets/enwiki_openwebtext_qwen3/merge`
  - `/share/datasets/pretrain/c4/llama3.1/c4_en_text_document`
  - `/share/datasets/pretrain/c4/llama3.1/Qwen2.5/c4_en_text_document`
- Tokenizer/reference model paths available:
  - `/share/models/llama-2-7b`
  - `/share/models/Llama-2-7B-Chat-hf`
  - `/share/models/Llama-3.1-8B-Instruct`
  - `/share/models/Qwen3-8B`
  - `/share/models/qwen3-8b`
- Downstream task cache available:
  - `/workspace/Megatron-LM-312/collected/tetrajet_v2_official/olmo2-training/olmo_data/hf_datasets`
  - includes `ai2_arc/ARC-Challenge`, `ai2_arc/ARC-Easy`, `hellaswag`, `piqa`, `sciq`, and `winogrande/winogrande_xl`

## Missing Or Not Yet Standardized

### 1. Unified Fprop-Input Paper Launcher

Current gap:

- `fprop-input` only has a single 100M ours-0.1% convergence script.
- The richer matrix launcher and variants are still in `/workspace/Megatron-LM-312`.

Need:

- Port or recreate a unified launcher in `fprop-input`, with variants:
  - `bf16`
  - `te_nvfp4`
  - `ours_0p1`
  - `quartet_ii`
- Make `ours_0p1` default to the current validated fast path:
  - `normal_threshold`
  - ratio `0.001`
  - `sum_then_add`
  - `CACHE_WEIGHT_T=0`
- Keep all variants under one W&B project per model family.

### 2. LLaMA2-7B 2B-Token Convergence Scripts

Current gap:

- No final fprop-input launcher for LLaMA2-7B 2B tokens.
- Existing LLaMA3-8B-shape script is a throughput/convergence probe, not the AGoQ-aligned LLaMA2-7B main run.

Need:

- Add LLaMA2-7B model config script.
- Compute and fix `TRAIN_ITERS = 2B / (global_batch_size * seq_length)`.
- Decide exact sequence length and parallelism on 8x RTX PRO 6000:
  - likely TP/PP/DP setup must be tested with a short smoke run.
- Enable checkpoint saving at the end or at coarse intervals for downstream eval.
- Run variants:
  - BF16
  - TE NVFP4
  - ours 0.1%
  - Quartet-II last

### 3. Second Small Model

Current gap:

- 100M 6k run is active.
- A second smaller model has not been fixed for the AAAI table.

Candidate choices:

- 560M Nanochat-like: cheaper, good for convergence sanity.
- 1.7B Qwen3-like or 1.9B Nanochat-like: closer to a meaningful mid-size point, more expensive.

Need:

- Pick one.
- Port corresponding script to `fprop-input`.
- Run BF16/TE/ours; Quartet-II can remain last.

### 4. Downstream Evaluation Chain

Current gap:

- Downstream datasets are present, but `transformer_engine` env is missing:
  - `lm_eval`
  - `datasets`
  - `accelerate`
- No one-command script currently converts final Megatron checkpoints to an eval-ready HF model and runs the six AGoQ tasks.

Need:

- Install or create an isolated eval environment.
- Add checkpoint export/convert command.
- Add evaluation wrapper for:
  - `arc_challenge`
  - `arc_easy`
  - `hellaswag`
  - `piqa`
  - `sciq`
  - `winogrande`
- Decide whether to use lm-evaluation-harness or the existing OLMo/TetraJet evaluation stack.
- Save JSON/CSV summaries with run name, checkpoint path, and W&B URL.

### 5. Throughput/System Script Matrix

Current gap:

- There are older TP2 8B-shape throughput scripts and logs.
- They do not yet encode the current `sum_then_add + no weight_t cache` version as the standard ours path.

Need:

- Add short-run throughput scripts for BF16, TE NVFP4, ours 0.1%, Quartet-II.
- Use benchmark-fast mode for system comparisons.
- Standardize warmup and measurement windows.
- Log:
  - median and mean step time
  - tokens/sec or samples/sec
  - TFLOP/s/GPU
  - memory peak
  - sparse correction backend/variant from logs

### 6. Result Collection And Reproducibility

Current gap:

- W&B is enabled, but no local result manifest ties together scripts, git state, data prefix, checkpoint, and downstream scores.

Need:

- Add a small result collection script that extracts:
  - final train/validation loss
  - mean/median step time after warmup
  - skipped/nan iterations
  - W&B run id
  - checkpoint path
- Add a per-run manifest in JSONL or CSV.
- Capture git diff or commit hash for both:
  - `/workspace/Megatron-LM-312-fprop-input`
  - `/workspace/Megatron-LM-312/collected/nvfp4_warpgroup_sparse_fusion`

## Suggested Immediate Order

1. Let the current 100M 6k run finish and compare its curve with the old base run.
2. Port the unified variant launcher into `fprop-input`.
3. Add a short smoke matrix for 100M: BF16, TE NVFP4, ours 0.1%, Quartet-II.
4. Add LLaMA2-7B 2B-token launcher and run a 50-100 step smoke for BF16/TE/ours.
5. Prepare the downstream eval environment and checkpoint conversion before starting the full 7B run.
6. Run full LLaMA2-7B 2B-token convergence.
7. Run downstream eval on final checkpoints.
8. Run throughput/system matrix, then Quartet-II.
