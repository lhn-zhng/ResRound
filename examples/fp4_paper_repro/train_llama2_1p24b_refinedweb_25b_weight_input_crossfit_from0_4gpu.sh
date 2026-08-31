#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
TRAIN_SCRIPT="${SCRIPT_DIR}/train_nanochat_1p9b_fineweb_edu.sh"

TRAIN_TOKENS="${TRAIN_TOKENS:-25000000000}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-${REPO_ROOT}/runs/llama2_1p24b_refinedweb_32k_tok25000000000}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/llama2_1p24b_refinedweb_32k_tok${TRAIN_TOKENS}_weight_input_r0p003_crossfit_from0}"
RUN_VARIANTS="${RUN_VARIANTS:-weight_input_r0p003_crossfit}"
DEST_VARIANT="${DEST_VARIANT:-${RUN_VARIANTS%%,*}}"
DEST_CHECKPOINT_DIR="${RUN_ROOT}/checkpoints/${DEST_VARIANT}"

export MEGATRON_CONDA_ENV="${MEGATRON_CONDA_ENV:-transformer_engine}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0a}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export MASTER_PORT="${MASTER_PORT:-6464}"
export WANDB_MODE="${WANDB_MODE:-online}"

export PLAN_LABEL="${PLAN_LABEL:-Llama-2-like 1.244B RefinedWeb-32K input-0.003 plus latest weight cross-fit from scratch}"
export MODEL_NAME="${MODEL_NAME:-llama2_1p24b_refinedweb_32k_tok${TRAIN_TOKENS}_weight_input_r0p003_crossfit_from0}"
export RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-${MODEL_NAME}}"
export RUN_VARIANTS
export WANDB_PROJECT="${WANDB_PROJECT:-megatron_fp4_llama2_1p24b}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-llama2_1p24b_weight_input_r0p003_crossfit_from0_s${SEED:-1234}_tok${TRAIN_TOKENS}}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"

export NUM_LAYERS="${NUM_LAYERS:-22}"
export HIDDEN_SIZE="${HIDDEN_SIZE:-2048}"
export FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-5504}"
export NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-16}"
export NUM_QUERY_GROUPS="${NUM_QUERY_GROUPS:-16}"
export KV_CHANNELS="${KV_CHANNELS:-128}"
export USE_GQA="${USE_GQA:-0}"
export USE_SWIGLU="${USE_SWIGLU:-1}"
export USE_SQUARED_RELU="${USE_SQUARED_RELU:-0}"
export QK_LAYERNORM="${QK_LAYERNORM:-0}"

export SEQ_LENGTH="${SEQ_LENGTH:-2048}"
export MAX_POSITION_EMBEDDINGS="${MAX_POSITION_EMBEDDINGS:-2048}"
export ROTARY_BASE="${ROTARY_BASE:-10000}"
export ROTARY_PERCENT="${ROTARY_PERCENT:-1.0}"
export NORMALIZATION="${NORMALIZATION:-RMSNorm}"
export NORM_EPSILON="${NORM_EPSILON:-1e-5}"
export VOCAB_SIZE="${VOCAB_SIZE:-32000}"
export MAKE_VOCAB_SIZE_DIVISIBLE_BY="${MAKE_VOCAB_SIZE_DIVISIBLE_BY:-128}"

export DATA_PATH="${DATA_PATH:-/share/datasets/pretrain/refinedweb/tmp/jetmoe_refinedweb_content_document}"
export TOKENIZER_MODEL="${TOKENIZER_MODEL:-/share/models/pretrain/jetmoe-8b}"
export REQUIRE_FINEWEB_COMPLETE=0
export DATA_CACHE_PATH="${DATA_CACHE_PATH:-${SOURCE_RUN_ROOT}/cache}"

export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-16}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
export TP_SIZE="${TP_SIZE:-1}"
export PP_SIZE="${PP_SIZE:-1}"
export CP_SIZE="${CP_SIZE:-1}"
export ENABLE_SEQUENCE_PARALLEL="${ENABLE_SEQUENCE_PARALLEL:-0}"

export OPTIMIZER="${OPTIMIZER:-adam}"
export LR="${LR:-3e-4}"
export MIN_LR="${MIN_LR:-3e-5}"
export LR_DECAY_STYLE="${LR_DECAY_STYLE:-cosine}"
export ADAM_BETA1="${ADAM_BETA1:-0.9}"
export ADAM_BETA2="${ADAM_BETA2:-0.95}"
export ADAM_EPS="${ADAM_EPS:-1e-8}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
export CLIP_GRAD="${CLIP_GRAD:-1.0}"
export SEED="${SEED:-1234}"

export FP4_WEIGHT_INPUT_RATIO="${FP4_WEIGHT_INPUT_RATIO:-0.003}"
export TRAIN_TOKENS
export SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-1}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
export EVAL_ITERS="${EVAL_ITERS:-10}"
export RUN_ROOT
export LOG_DIR="${LOG_DIR:-${RUN_ROOT}/launchers}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUN_ROOT}/checkpoints}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-${RUN_ROOT}/tensorboard}"

extra_args=(
    --cross-entropy-fusion-impl te
    --empty-unused-memory-level 0
    --log-memory-interval 10
    --auto-detect-ckpt-format
    --save-retain-interval 250000000
)
if [[ -s "${DEST_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" ]]; then
    if [[ "${RESUME_EXISTING_CHECKPOINT:-0}" != 1 ]]; then
        echo "Destination already contains a checkpoint; refusing a from-zero launch: ${DEST_CHECKPOINT_DIR}" >&2
        echo "Set RESUME_EXISTING_CHECKPOINT=1 to resume it explicitly." >&2
        exit 2
    fi
    extra_args+=(--load "${DEST_CHECKPOINT_DIR}")
    echo "[llama2-1p24b-crossfit] explicit resume ${DEST_CHECKPOINT_DIR}"
fi
printf -v EXTRA_MEGATRON_ARGS_JOINED '%q ' "${extra_args[@]}"
USER_EXTRA_MEGATRON_ARGS="${EXTRA_MEGATRON_ARGS:-}"
export EXTRA_MEGATRON_ARGS="${EXTRA_MEGATRON_ARGS_JOINED% }${USER_EXTRA_MEGATRON_ARGS:+ ${USER_EXTRA_MEGATRON_ARGS}}"

exec bash "${TRAIN_SCRIPT}"
