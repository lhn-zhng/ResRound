#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
TRAIN_TOKENS="${TRAIN_TOKENS:-25000000000}"
RUN_TAG="input0p003_currentcode_from0"

# Strict from-zero counterpart for the joint-residual long run.  This uses
# the same current worktree, seed, data order, optimizer, and input-outlier
# recipe, while leaving ephemeral weight rounding disabled.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export MASTER_PORT="${MASTER_PORT:-6531}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-megatron_fp4_llama2_1p24b}"

export RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/llama2_1p24b_refinedweb_32k_tok${TRAIN_TOKENS}_${RUN_TAG}}"
export MODEL_NAME="${MODEL_NAME:-llama2_1p24b_refinedweb_32k_tok${TRAIN_TOKENS}_${RUN_TAG}}"
export RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-${MODEL_NAME}}"
export RUN_VARIANTS=our
export DEST_VARIANT=our
export FP4_OUTLIER_RATIO=0.003
export PLAN_LABEL="Llama-2-like 1.244B RefinedWeb-32K current-code input-0.003 control from scratch"
export WANDB_RUN_ID="${WANDB_RUN_ID:-llama2_1p24b_${RUN_TAG}_s${SEED:-1234}_tok${TRAIN_TOKENS}}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"

exec bash "${SCRIPT_DIR}/train_llama2_1p24b_refinedweb_25b_weight_input_crossfit_from0_4gpu.sh"
