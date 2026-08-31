#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
TRAIN_SCRIPT="${SCRIPT_DIR}/train_nanochat_1p9b_fineweb_edu.sh"

TRAIN_TOKENS="${TRAIN_TOKENS:-25000000000}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-${REPO_ROOT}/runs/llama2_1p24b_refinedweb_32k_tok25000000000}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/llama1p24b_bvr_qkv8_proj22_from0}"
DEST_CHECKPOINT_DIR="${RUN_ROOT}/checkpoints/weight_input_r0p003"
EXIT_INTERVAL="${EXIT_INTERVAL:-1000}"

export MEGATRON_CONDA_ENV="${MEGATRON_CONDA_ENV:-transformer_engine}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0a}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export MASTER_PORT="${MASTER_PORT:-6501}"
export WANDB_MODE="${WANDB_MODE:-online}"

export PLAN_LABEL="Llama-2-like 1.244B RefinedWeb-32K BVR QKV[0,8)+Proj[0,22) from scratch"
export MODEL_NAME="llama1p24b_bvr_qkv8_proj22_from0"
export RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-${MODEL_NAME}}"
export RUN_VARIANTS="${RUN_VARIANTS:-weight_input_r0p003}"
export WANDB_PROJECT="${WANDB_PROJECT:-megatron_fp4_llama2_1p24b}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-llama1p24b_bvr_qkv8_proj22_from0_s${SEED:-1234}}"
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
export FP4_WEIGHT_ROUNDING_GRAD_ALIGNMENT=1
export FP4_WEIGHT_ROUNDING_GRAD_ALIGNMENT_CAPTURE_GRAD=1
export FP4_WEIGHT_ROUNDING_GRAD_ALIGNMENT_ROW_PREDICT=1
export FP4_WEIGHT_ROUNDING_GRAD_ALIGNMENT_ROW_TOKENS="${BVR_FEEDBACK_TOKENS:-128}"
export FP4_WEIGHT_ROUNDING_GRAD_ALIGNMENT_MAX_HARM_FRACTION="${BVR_MAX_HARM_FRACTION:-0.5}"
export FP4_WEIGHT_ROUNDING_GRAD_ALIGNMENT_ROW_APPLY=1
export FP4_WEIGHT_ROUNDING_GRAD_ALIGNMENT_ROW_APPLY_SLOTS="${BVR_MODULE_SLOTS:-0,1}"
export FP4_WEIGHT_ROUNDING_GRAD_ALIGNMENT_ROW_EVAL_INPUT_ONLY=1
export FP4_WEIGHT_ROUNDING_GRAD_ALIGNMENT_LOG="${BVR_VERBOSE_ALIGNMENT_LOG:-0}"
export FP4_WEIGHT_ROUNDING_GRAD_ALIGNMENT_MODULES="${BVR_MODULE_IDS:-$(seq -s, 0 65)}"

export TRAIN_TOKENS
export SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-1}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-200}"
export EVAL_ITERS="${EVAL_ITERS:-10}"
export RUN_ROOT
export LOG_DIR="${LOG_DIR:-${RUN_ROOT}/launchers}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUN_ROOT}/checkpoints}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-${RUN_ROOT}/tensorboard}"

if [[ -s "${DEST_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" ]]; then
    echo "Destination already contains a checkpoint; refusing a from-zero launch: ${DEST_CHECKPOINT_DIR}" >&2
    exit 2
fi

extra_args=(
    --cross-entropy-fusion-impl te
    --empty-unused-memory-level 0
    --log-memory-interval 10
    --auto-detect-ckpt-format
    --save-retain-interval 250000000
    --fp4-outlier-weight-rounding-qkv-layer-end 8
    --fp4-outlier-weight-rounding-proj-layer-end -1
    --fp4-outlier-weight-rounding-fc1-layer-end 0
)
if (( EXIT_INTERVAL > 0 )); then
    extra_args+=(--exit-interval "${EXIT_INTERVAL}")
fi
printf -v EXTRA_MEGATRON_ARGS_JOINED '%q ' "${extra_args[@]}"
USER_EXTRA_MEGATRON_ARGS="${EXTRA_MEGATRON_ARGS:-}"
export EXTRA_MEGATRON_ARGS="${EXTRA_MEGATRON_ARGS_JOINED% }${USER_EXTRA_MEGATRON_ARGS:+ ${USER_EXTRA_MEGATRON_ARGS}}"

exec bash "${TRAIN_SCRIPT}"
