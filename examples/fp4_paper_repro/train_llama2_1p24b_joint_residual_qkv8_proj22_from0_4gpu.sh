#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
TRAIN_TOKENS="${TRAIN_TOKENS:-25000000000}"
RUN_TAG="joint_residual_r0p003_p2048_a1024_r0p50_qkv8_proj22_continuous_from0_20260725"

# Continuous from-zero run used for the final comparison against the completed
# input-only baseline.  Do not add intermediate evaluations, segmented exits,
# or checkpoint loads: those would change the validation sample stream and the
# stochastic training trajectory.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export MASTER_PORT="${MASTER_PORT:-6530}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-megatron_fp4_llama2_1p24b}"

# selection_tokens is the total sample.  Cross-fit uses 2048 proposal tokens
# and 1024 disjoint audit tokens on each rank and microbatch.
export FP4_WEIGHT_ROUNDING_SELECTION_TOKENS=3072
export FP4_WEIGHT_ROUNDING_AUDIT_TOKENS=1024
export FP4_WEIGHT_ROUNDING_AUDIT_MAX_REGRESSION_FRACTION=0.50
export FP4_WEIGHT_ROUNDING_ROUNDS_PER_GROUP=2
export FP4_WEIGHT_ROUNDING_OFFDIAG_SHRINK=0.5
export FP4_WEIGHT_ROUNDING_EVAL_INPUT_ONLY=1

export RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/llama2_1p24b_refinedweb_32k_tok${TRAIN_TOKENS}_${RUN_TAG}}"
export RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-llama2_1p24b_refinedweb_32k_tok${TRAIN_TOKENS}_${RUN_TAG}}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-llama2_1p24b_${RUN_TAG}_s${SEED:-1234}_tok${TRAIN_TOKENS}}"
export WANDB_RESUME="${WANDB_RESUME:-never}"

joint_args=(
    --fp4-outlier-weight-rounding-joint-objective
    --fp4-outlier-weight-rounding-combined-audit
    --fp4-outlier-weight-rounding-layer-start 0
    --fp4-outlier-weight-rounding-qkv-layer-end 8
    --fp4-outlier-weight-rounding-proj-layer-end -1
    --fp4-outlier-weight-rounding-fc1-layer-end 0
)
printf -v JOINT_ARGS_JOINED '%q ' "${joint_args[@]}"
USER_EXTRA_MEGATRON_ARGS="${EXTRA_MEGATRON_ARGS:-}"
export EXTRA_MEGATRON_ARGS="${JOINT_ARGS_JOINED% }${USER_EXTRA_MEGATRON_ARGS:+ ${USER_EXTRA_MEGATRON_ARGS}}"

exec bash "${SCRIPT_DIR}/train_llama2_1p24b_refinedweb_25b_weight_input_crossfit_from0_4gpu.sh"
