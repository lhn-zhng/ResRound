#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
    source /opt/conda/etc/profile.d/conda.sh
    if [[ "${CONDA_DEFAULT_ENV:-}" != "transformer_engine" ]]; then
        conda activate transformer_engine
    fi
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

TE_SITE_PACKAGES="${CONDA_PREFIX:-/opt/conda/envs/transformer_engine}/lib/python3.12/site-packages"
export LD_LIBRARY_PATH="${TE_SITE_PACKAGES}/nvidia/cudnn/lib:${TE_SITE_PACKAGES}/nvidia/cublas/lib:${TE_SITE_PACKAGES}/nvidia/cuda_runtime/lib:${TE_SITE_PACKAGES}/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-12.0a}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_INIT_TIMEOUT=${WANDB_INIT_TIMEOUT:-300}

CHECKPOINT_PATH=${CHECKPOINT_PATH:-"checkpoints/llama3_8b_fprop_input_outlier_r003_ag"}
TENSORBOARD_LOGS_PATH=${TENSORBOARD_LOGS_PATH:-"tensorboard_logs/llama3_8b_fprop_input_outlier_r003_ag"}
DEFAULT_TOKENIZER_ARG="/share/models/qwen3-8b"
DEFAULT_DATA_ARG="${REPO_ROOT}/datasets/llama3/merge"
if [[ ! -f "${DEFAULT_DATA_ARG}.idx" || ! -f "${DEFAULT_DATA_ARG}.bin" ]]; then
    if [[ -f "/workspace/Megatron-LM-312/datasets/llama3/merge.idx" && -f "/workspace/Megatron-LM-312/datasets/llama3/merge.bin" ]]; then
        DEFAULT_DATA_ARG="/workspace/Megatron-LM-312/datasets/llama3/merge"
    else
        DEFAULT_DATA_ARG="MOCK"
        DEFAULT_TOKENIZER_ARG="MOCK"
    fi
fi
TOKENIZER_ARG=${TOKENIZER_ARG:-"${DEFAULT_TOKENIZER_ARG}"}
DATA_ARG=${DATA_ARG:-"${DEFAULT_DATA_ARG}"}
LOG_DIR=${LOG_DIR:-"${REPO_ROOT}/logs"}
mkdir -p "$(dirname "${CHECKPOINT_PATH}")" "$(dirname "${TENSORBOARD_LOGS_PATH}")" "${LOG_DIR}"

GPUS_PER_NODE=${GPUS_PER_NODE:-2}
NUM_NODES=${NUM_NODES:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
NODE_RANK=${NODE_RANK:-0}
if [[ -z "${MASTER_PORT:-}" ]]; then
    if [[ "${NUM_NODES}" == "1" ]]; then
        MASTER_PORT=$(python - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("", 0))
    print(sock.getsockname()[1])
PY
)
    else
        MASTER_PORT=6027
    fi
fi

PRETRAIN_SCRIPT_PATH="${REPO_ROOT}/pretrain_gpt.py"

# Same model shape as train_llama3_8b_h100_fp4_nccl_6_new_r003_adaptive.sh.
TP_SIZE=${TP_SIZE:-2}
CP_SIZE=${CP_SIZE:-1}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}
NUM_LAYERS=${NUM_LAYERS:-8}
SEQ_LENGTH=${SEQ_LENGTH:-4096}
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-8192}
HIDDEN_SIZE=${HIDDEN_SIZE:-4096}
FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-14336}
NUM_ATTENTION_HEADS=${NUM_ATTENTION_HEADS:-32}
NUM_QUERY_GROUPS=${NUM_QUERY_GROUPS:-8}
KV_CHANNELS=${KV_CHANNELS:-128}

TRAIN_STEPS=${TRAIN_STEPS:-50000}
LOG_INTERVAL=${LOG_INTERVAL:-1}
EVAL_ITERS=${EVAL_ITERS:-32}
EVAL_INTERVAL=${EVAL_INTERVAL:-100}
SKIP_EVAL=${SKIP_EVAL:-0}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000000}
SAVE_CHECKPOINT=${SAVE_CHECKPOINT:-0}
TENSORBOARD_LOG_INTERVAL=${TENSORBOARD_LOG_INTERVAL:-10}
ENABLE_PROFILE=${ENABLE_PROFILE:-0}
PROFILE_STEP_START=${PROFILE_STEP_START:-4}
PROFILE_STEP_END=${PROFILE_STEP_END:-6}
BENCHMARK_FAST_MODE=${BENCHMARK_FAST_MODE:-0}
if [[ "${BENCHMARK_FAST_MODE}" == "1" ]]; then
    ENABLE_MANUAL_GC=${ENABLE_MANUAL_GC:-0}
    EMPTY_UNUSED_MEMORY_LEVEL=${EMPTY_UNUSED_MEMORY_LEVEL:-0}
    CHECK_FOR_NAN_IN_LOSS_AND_GRAD=${CHECK_FOR_NAN_IN_LOSS_AND_GRAD:-0}
    LOG_MEMORY_TO_TENSORBOARD=${LOG_MEMORY_TO_TENSORBOARD:-0}
    LOG_TIMERS_TO_TENSORBOARD=${LOG_TIMERS_TO_TENSORBOARD:-0}
    LOG_VALIDATION_PPL_TO_TENSORBOARD=${LOG_VALIDATION_PPL_TO_TENSORBOARD:-0}
else
    ENABLE_MANUAL_GC=${ENABLE_MANUAL_GC:-1}
    EMPTY_UNUSED_MEMORY_LEVEL=${EMPTY_UNUSED_MEMORY_LEVEL:-1}
    CHECK_FOR_NAN_IN_LOSS_AND_GRAD=${CHECK_FOR_NAN_IN_LOSS_AND_GRAD:-1}
    LOG_MEMORY_TO_TENSORBOARD=${LOG_MEMORY_TO_TENSORBOARD:-1}
    LOG_TIMERS_TO_TENSORBOARD=${LOG_TIMERS_TO_TENSORBOARD:-1}
    LOG_VALIDATION_PPL_TO_TENSORBOARD=${LOG_VALIDATION_PPL_TO_TENSORBOARD:-1}
fi
LOG_WORLD_SIZE_TO_TENSORBOARD=${LOG_WORLD_SIZE_TO_TENSORBOARD:-1}

FP4_QUANTIZER_FACTORY=${FP4_QUANTIZER_FACTORY:-megatron.core.extensions.fp4_outlier_recipe.nvfp4_outlier_quantizer_factory}
FP4_TE_CONFIG=${FP4_TE_CONFIG:-"${REPO_ROOT}/megatron/core/extensions/fp4_outlier_config.yaml"}
FP4_RECIPE=${FP4_RECIPE:-custom}
FP4_OUTLIER_RATIO=${FP4_OUTLIER_RATIO:-0.02}
FP4_OUTLIER_SELECTION_METHOD=${FP4_OUTLIER_SELECTION_METHOD:-normal_threshold}
FP4_OUTLIER_ADAPTIVE_RATIO=${FP4_OUTLIER_ADAPTIVE_RATIO:-1}
FP4_OUTLIER_ADAPTIVE_MIN_RATIO=${FP4_OUTLIER_ADAPTIVE_MIN_RATIO:-0.0}
FP4_OUTLIER_ADAPTIVE_MAX_RATIO=${FP4_OUTLIER_ADAPTIVE_MAX_RATIO:-0.05}
FP4_OUTLIER_ADAPTIVE_REFERENCE_HEAVINESS=${FP4_OUTLIER_ADAPTIVE_REFERENCE_HEAVINESS:-15.0}
FP4_OUTLIER_ENABLE_FPROP=${FP4_OUTLIER_ENABLE_FPROP:-1}
FP4_OUTLIER_ENABLE_FAST_FPROP=${FP4_OUTLIER_ENABLE_FAST_FPROP:-0}
FP4_OUTLIER_ENABLE_NVFP4_A1_A2_ALL_GATHER=${FP4_OUTLIER_ENABLE_NVFP4_A1_A2_ALL_GATHER:-1}
FP4_OUTLIER_STORE_INPUT_DENSE_MAIN=${FP4_OUTLIER_STORE_INPUT_DENSE_MAIN:-0}
FP4_OUTLIER_MAIN_QUANTIZER_RHT=${FP4_OUTLIER_MAIN_QUANTIZER_RHT:-0}
FP4_OUTLIER_INPUT_STOCHASTIC_ROUNDING=${FP4_OUTLIER_INPUT_STOCHASTIC_ROUNDING:-0}

if [[ "${SKIP_EVAL}" == "1" ]]; then
    EVAL_ITERS=0
    EVAL_INTERVAL=1000000
fi

DATA_CACHE_PATH=${DATA_CACHE_PATH:-"${REPO_ROOT}/benchmark_cache_llama3_8b_fprop_input_outlier"}
mkdir -p "${DATA_CACHE_PATH}"

if [[ ! -f "${PRETRAIN_SCRIPT_PATH}" ]]; then
    echo "Error: pretrain_gpt.py not found at ${PRETRAIN_SCRIPT_PATH}" >&2
    exit 1
fi

DISTRIBUTED_ARGS=(
    --nproc_per_node "${GPUS_PER_NODE}"
    --nnodes "${NUM_NODES}"
    --node_rank "${NODE_RANK}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
)

MODEL_ARGS=(
    --use-mcore-models
    --num-layers "${NUM_LAYERS}"
    --hidden-size "${HIDDEN_SIZE}"
    --ffn-hidden-size "${FFN_HIDDEN_SIZE}"
    --num-attention-heads "${NUM_ATTENTION_HEADS}"
    --group-query-attention
    --num-query-groups "${NUM_QUERY_GROUPS}"
    --kv-channels "${KV_CHANNELS}"
    --seq-length "${SEQ_LENGTH}"
    --max-position-embeddings "${MAX_POSITION_EMBEDDINGS}"
    --position-embedding-type rope
    --rotary-base 1000000
    --rotary-percent 1.0
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --swiglu
    --normalization RMSNorm
    --init-method-std 0.0134
    --attention-backend fused
    --apply-layernorm-1p
    --untie-embeddings-and-output-weights
    --disable-bias-linear
)

TRAINING_ARGS=(
    --micro-batch-size "${MICRO_BATCH_SIZE}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --train-samples "$((TRAIN_STEPS * GLOBAL_BATCH_SIZE))"
    --lr-decay-samples 1949218748
    --lr-warmup-samples 3906252
    --lr 0.00015
    --min-lr 0.00001
    --decoupled-lr 5.0e-4
    --decoupled-min-lr 4.5e-5
    --lr-decay-style cosine
    --clip-grad 1.0
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --bf16
    --grad-reduce-in-bf16
    --cross-entropy-loss-fusion
    --calculate-per-token-loss
    --empty-unused-memory-level "${EMPTY_UNUSED_MEMORY_LEVEL}"
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)
if [[ "${ENABLE_MANUAL_GC}" == "1" ]]; then
    TRAINING_ARGS+=(--manual-gc)
fi
if [[ "${CHECK_FOR_NAN_IN_LOSS_AND_GRAD}" == "0" ]]; then
    TRAINING_ARGS+=(--no-check-for-nan-in-loss-and-grad)
fi

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size "${TP_SIZE}"
    --context-parallel-size "${CP_SIZE}"
    --sequence-parallel
)

FP4_ARGS=()

if [[ "${FP4_RECIPE}" != "bf16" ]]; then
    FP4_ARGS=(
        --fp4-format e2m1
        --fp4-recipe "${FP4_RECIPE}"
    )
fi

if [[ "${FP4_RECIPE}" == "custom" ]]; then
    FP4_ARGS+=(
        --te-precision-config-file "${FP4_TE_CONFIG}"
        --fp4-quantizer-factory "${FP4_QUANTIZER_FACTORY}"
        --fp4-outlier-ratio "${FP4_OUTLIER_RATIO}"
        --fp4-outlier-selection-method "${FP4_OUTLIER_SELECTION_METHOD}"
        --no-fp4-outlier-enable-dgrad
        --no-fp4-outlier-enable-wgrad
    )

    if [[ "${FP4_OUTLIER_ADAPTIVE_RATIO}" == "1" ]]; then
        FP4_ARGS+=(
            --fp4-outlier-adaptive-ratio
            --fp4-outlier-adaptive-min-ratio "${FP4_OUTLIER_ADAPTIVE_MIN_RATIO}"
            --fp4-outlier-adaptive-max-ratio "${FP4_OUTLIER_ADAPTIVE_MAX_RATIO}"
            --fp4-outlier-adaptive-reference-heaviness "${FP4_OUTLIER_ADAPTIVE_REFERENCE_HEAVINESS}"
        )
    fi
    if [[ "${FP4_OUTLIER_ENABLE_FPROP}" == "0" ]]; then
        FP4_ARGS+=(--no-fp4-outlier-enable-fprop)
    fi
    if [[ "${FP4_OUTLIER_ENABLE_FAST_FPROP}" == "1" ]]; then
        FP4_ARGS+=(--fp4-outlier-enable-fast-fprop)
    fi
    if [[ "${FP4_OUTLIER_ENABLE_NVFP4_A1_A2_ALL_GATHER}" == "1" ]]; then
        FP4_ARGS+=(--fp4-outlier-enable-nvfp4-a1-a2-all-gather)
    fi
    if [[ "${FP4_OUTLIER_STORE_INPUT_DENSE_MAIN}" == "1" ]]; then
        FP4_ARGS+=(--fp4-outlier-store-input-dense-main)
    fi
    if [[ "${FP4_OUTLIER_MAIN_QUANTIZER_RHT}" == "1" ]]; then
        FP4_ARGS+=(--fp4-outlier-main-quantizer-rht)
    fi
    if [[ "${FP4_OUTLIER_INPUT_STOCHASTIC_ROUNDING}" == "1" ]]; then
        FP4_ARGS+=(--fp4-outlier-input-stochastic-rounding)
    fi
elif [[ "${FP4_RECIPE}" != "nvfp4" && "${FP4_RECIPE}" != "bf16" ]]; then
    echo "Error: FP4_RECIPE must be 'custom', 'nvfp4', or 'bf16', got '${FP4_RECIPE}'." >&2
    exit 1
fi

DATA_ARGS=()
if [[ "${TOKENIZER_ARG}" == "MOCK" || "${DATA_ARG}" == "MOCK" || -z "${TOKENIZER_ARG}" ]]; then
    DATA_ARGS=(
        --mock-data
        --tokenizer-type NullTokenizer
        --vocab-size 128256
        --data-cache-path "${DATA_CACHE_PATH}"
        --tiktoken-pattern v2
        --split 99,1,0
        --no-create-attention-mask-in-dataloader
        --no-mmap-bin-files
        --num-workers 1
    )
else
    if [[ ! -f "${DATA_ARG}.idx" || ! -f "${DATA_ARG}.bin" ]]; then
        echo "Error: data prefix '${DATA_ARG}' is missing .idx or .bin files." >&2
        echo "Set DATA_ARG=/path/to/prefix, or use DATA_ARG=MOCK TOKENIZER_ARG=MOCK for a smoke run." >&2
        exit 1
    fi
    if [[ ! -e "${TOKENIZER_ARG}" ]]; then
        echo "Error: tokenizer path '${TOKENIZER_ARG}' does not exist." >&2
        echo "Set TOKENIZER_ARG=/path/to/tokenizer, or use DATA_ARG=MOCK TOKENIZER_ARG=MOCK for a smoke run." >&2
        exit 1
    fi
    DATA_ARGS=(
        --data-path "${DATA_ARG}"
        --tokenizer-type HuggingFaceTokenizer
        --tokenizer-model "${TOKENIZER_ARG}"
        --data-cache-path "${DATA_CACHE_PATH}"
        --split 99,1,0
        --no-create-attention-mask-in-dataloader
        --no-mmap-bin-files
        --num-workers 1
        --vocab-size 128256
    )
fi

LOGGING_ARGS=(
    --log-interval "${LOG_INTERVAL}"
    --eval-iters "${EVAL_ITERS}"
    --eval-interval "${EVAL_INTERVAL}"
    --log-throughput
    --distributed-timeout-minutes 60
    --tensorboard-dir "${TENSORBOARD_LOGS_PATH}"
    --tensorboard-queue-size 1000
    --tensorboard-log-interval "${TENSORBOARD_LOG_INTERVAL}"
)
if [[ "${LOG_MEMORY_TO_TENSORBOARD}" == "1" ]]; then
    LOGGING_ARGS+=(--log-memory-to-tensorboard)
fi
if [[ "${LOG_TIMERS_TO_TENSORBOARD}" == "1" ]]; then
    LOGGING_ARGS+=(--log-timers-to-tensorboard)
fi
if [[ "${LOG_VALIDATION_PPL_TO_TENSORBOARD}" == "1" ]]; then
    LOGGING_ARGS+=(--log-validation-ppl-to-tensorboard)
fi
if [[ "${LOG_WORLD_SIZE_TO_TENSORBOARD}" == "1" ]]; then
    LOGGING_ARGS+=(--log-world-size-to-tensorboard)
fi

if [[ "${WANDB_MODE}" != "disabled" ]]; then
    WANDB_PROJECT_NAME=${WANDB_PROJECT:-megatron_fp4}
    WANDB_EXP_NAME=${WANDB_EXP_NAME:-fp4_${FP4_RECIPE}_fast${FP4_OUTLIER_ENABLE_FAST_FPROP}_r${FP4_OUTLIER_RATIO}_ag-gpu${GPUS_PER_NODE}-tp${TP_SIZE}-seq${SEQ_LENGTH}}
    LOGGING_ARGS+=(
        --wandb-exp-name "${WANDB_EXP_NAME}"
        --wandb-project "${WANDB_PROJECT_NAME}"
        --wandb-save-dir "${LOG_DIR}/wandb_fp4"
    )
fi

if [[ "${SAVE_CHECKPOINT}" == "1" ]]; then
    LOGGING_ARGS+=(
        --save-interval "${SAVE_INTERVAL}"
        --ckpt-format torch_dist
        --save "${CHECKPOINT_PATH}"
    )
fi

if [[ "${ENABLE_PROFILE}" == "1" ]]; then
    LOGGING_ARGS+=(
        --profile
        --profile-step-start "${PROFILE_STEP_START}"
        --profile-step-end "${PROFILE_STEP_END}"
    )
fi

RUN_LOG="${LOG_DIR}/train_fp4_fprop_input_outlier_r003_ag.log"

echo "========== FP4 FPROP Input Outlier R003 AG =========="
echo "REPO_ROOT=${REPO_ROOT}"
echo "PYTHON=$(command -v python)"
echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-<unset>}"
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "GPUS_PER_NODE=${GPUS_PER_NODE} TP_SIZE=${TP_SIZE} sequence_parallel=1"
echo "TRAIN_STEPS=${TRAIN_STEPS} MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE} GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "EVAL_ITERS=${EVAL_ITERS} EVAL_INTERVAL=${EVAL_INTERVAL} SKIP_EVAL=${SKIP_EVAL}"
echo "NUM_LAYERS=${NUM_LAYERS} HIDDEN_SIZE=${HIDDEN_SIZE} SEQ_LENGTH=${SEQ_LENGTH}"
echo "BENCHMARK_FAST_MODE=${BENCHMARK_FAST_MODE}"
echo "ENABLE_MANUAL_GC=${ENABLE_MANUAL_GC} EMPTY_UNUSED_MEMORY_LEVEL=${EMPTY_UNUSED_MEMORY_LEVEL}"
echo "CHECK_FOR_NAN_IN_LOSS_AND_GRAD=${CHECK_FOR_NAN_IN_LOSS_AND_GRAD}"
echo "LOG_MEMORY_TO_TENSORBOARD=${LOG_MEMORY_TO_TENSORBOARD} LOG_TIMERS_TO_TENSORBOARD=${LOG_TIMERS_TO_TENSORBOARD}"
echo "TOKENIZER_ARG=${TOKENIZER_ARG}"
echo "DATA_ARG=${DATA_ARG}"
echo "WANDB_MODE=${WANDB_MODE}"
echo "WANDB_PROJECT=${WANDB_PROJECT_NAME:-<disabled>}"
echo "WANDB_EXP_NAME=${WANDB_EXP_NAME:-<disabled>}"
echo "FP4_RECIPE=${FP4_RECIPE}"
echo "FP4_OUTLIER_RATIO=${FP4_OUTLIER_RATIO}"
echo "FP4_OUTLIER_SELECTION_METHOD=${FP4_OUTLIER_SELECTION_METHOD}"
echo "FP4_OUTLIER_ADAPTIVE_RATIO=${FP4_OUTLIER_ADAPTIVE_RATIO}"
echo "FP4_OUTLIER_ADAPTIVE_MIN_RATIO=${FP4_OUTLIER_ADAPTIVE_MIN_RATIO}"
echo "FP4_OUTLIER_ADAPTIVE_MAX_RATIO=${FP4_OUTLIER_ADAPTIVE_MAX_RATIO}"
echo "FP4_OUTLIER_ADAPTIVE_REFERENCE_HEAVINESS=${FP4_OUTLIER_ADAPTIVE_REFERENCE_HEAVINESS}"
echo "FP4_OUTLIER_ENABLE_FAST_FPROP=${FP4_OUTLIER_ENABLE_FAST_FPROP}"
echo "FP4_OUTLIER_ENABLE_NVFP4_A1_A2_ALL_GATHER=${FP4_OUTLIER_ENABLE_NVFP4_A1_A2_ALL_GATHER}"
echo "FP4_TE_CONFIG=${FP4_TE_CONFIG}"
echo "LOG=${RUN_LOG}"
echo "====================================================="

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' torchrun "${DISTRIBUTED_ARGS[@]}" "${PRETRAIN_SCRIPT_PATH}" \
        "${MODEL_ARGS[@]}" "${TRAINING_ARGS[@]}" "${FP4_ARGS[@]}" \
        "${MODEL_PARALLEL_ARGS[@]}" "${DATA_ARGS[@]}" "${LOGGING_ARGS[@]}"
    printf '\n'
    exit 0
fi

torchrun "${DISTRIBUTED_ARGS[@]}" \
    "${PRETRAIN_SCRIPT_PATH}" \
    "${MODEL_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${FP4_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    2>&1 | tee "${RUN_LOG}"
