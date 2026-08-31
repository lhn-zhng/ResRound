#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
    source /opt/conda/etc/profile.d/conda.sh
    if [[ "${CONDA_DEFAULT_ENV:-}" != "${MEGATRON_CONDA_ENV:-transformer_engine}" ]]; then
        conda activate "${MEGATRON_CONDA_ENV:-transformer_engine}"
    fi
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

TE_SITE_PACKAGES="${CONDA_PREFIX:-/opt/conda/envs/transformer_engine}/lib/python3.12/site-packages"
export LD_LIBRARY_PATH="${TE_SITE_PACKAGES}/nvidia/cudnn/lib:${TE_SITE_PACKAGES}/nvidia/cublas/lib:${TE_SITE_PACKAGES}/nvidia/cuda_runtime/lib:${TE_SITE_PACKAGES}/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"

detect_gpu_count() {
    local count
    if command -v nvidia-smi >/dev/null 2>&1; then
        count=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
        if [[ "${count}" =~ ^[0-9]+$ ]] && (( count > 0 )); then
            echo "${count}"
            return
        fi
    fi
    echo 1
}

make_cuda_visible_devices() {
    local count=$1
    local max_count=${2:-8}
    local use_count=${count}
    local devices="0"
    local i

    if (( use_count > max_count )); then
        use_count=${max_count}
    fi
    for (( i = 1; i < use_count; i++ )); do
        devices="${devices},${i}"
    done
    echo "${devices}"
}

cuda_visible_device_count() {
    local devices=$1
    if [[ -z "${devices}" || "${devices}" == "-1" ]]; then
        echo 0
        return
    fi
    awk -F',' '{print NF}' <<< "${devices}"
}

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    DETECTED_GPUS=$(detect_gpu_count)
    CUDA_VISIBLE_DEVICES=$(make_cuda_visible_devices "${DETECTED_GPUS}" 8)
    export CUDA_VISIBLE_DEVICES
fi
DEFAULT_GPUS_PER_NODE=$(cuda_visible_device_count "${CUDA_VISIBLE_DEVICES}")
if (( DEFAULT_GPUS_PER_NODE < 1 )); then
    DEFAULT_GPUS_PER_NODE=1
fi
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0a}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"

# Fast FPROP input-outlier defaults validated on the fprop-input branch.
export FP4_OUTLIER_FAST_FPROP_DIRECT_SPARSE_VARIANT="${FP4_OUTLIER_FAST_FPROP_DIRECT_SPARSE_VARIANT:-sum_then_add}"
export FP4_OUTLIER_FAST_FPROP_CACHE_WEIGHT_T="${FP4_OUTLIER_FAST_FPROP_CACHE_WEIGHT_T:-0}"
export NVFP4_DIRECT_ADD_TMA_VARIANT="${NVFP4_DIRECT_ADD_TMA_VARIANT:-sumthenadd}"
export FP4_OUTLIER_SPARSE_CORRECTION_BACKEND="${FP4_OUTLIER_SPARSE_CORRECTION_BACKEND:-auto}"

MODEL_NAME="${MODEL_NAME:-gpt2_345m_openwebtext}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-${MODEL_NAME}}"
RUN_VARIANTS="${RUN_VARIANTS:-bf16,te,our}"
WANDB_PROJECT="${WANDB_PROJECT:-megatron_fp4_paper_repro_train_gpt2_345m}"

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/checkpoints/fp4_paper_repro/${MODEL_NAME}}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${REPO_ROOT}/tensorboard_logs/fp4_paper_repro/${MODEL_NAME}}"
DATA_CACHE_PATH="${DATA_CACHE_PATH:-${REPO_ROOT}/benchmark_cache_fp4_paper_repro/${MODEL_NAME}}"
mkdir -p "${LOG_DIR}" "${CHECKPOINT_DIR}" "${TENSORBOARD_DIR}" "${DATA_CACHE_PATH}" "${LOG_DIR}/wandb_fp4"

DATA_PATH="${DATA_PATH:-${REPO_ROOT}/datasets/openwebtext_gpt2/bpe_openwebtext}"
ALLOW_MOCK_DATA="${ALLOW_MOCK_DATA:-0}"

TOKENIZER_TYPE="${TOKENIZER_TYPE:-HuggingFaceTokenizer}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-gpt2}"
VOCAB_FILE="${VOCAB_FILE:-}"
MERGE_FILE="${MERGE_FILE:-}"
VOCAB_SIZE="${VOCAB_SIZE:-50257}"
MAKE_VOCAB_SIZE_DIVISIBLE_BY="${MAKE_VOCAB_SIZE_DIVISIBLE_BY:-128}"

GPUS_PER_NODE="${GPUS_PER_NODE:-${DEFAULT_GPUS_PER_NODE}}"
NUM_NODES="${NUM_NODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-6231}"
TP_SIZE="${TP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"

NUM_LAYERS="${NUM_LAYERS:-24}"
HIDDEN_SIZE="${HIDDEN_SIZE:-1024}"
FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-4096}"
NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-16}"
SEQ_LENGTH="${SEQ_LENGTH:-1024}"
MAX_POSITION_EMBEDDINGS="${MAX_POSITION_EMBEDDINGS:-1024}"

MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-32}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-512}"
DEFAULT_TRAIN_TOKENS="${DEFAULT_TRAIN_TOKENS:-9000000000}"
TOKENS_PER_STEP=$((GLOBAL_BATCH_SIZE * SEQ_LENGTH))
if [[ -n "${TRAIN_TOKENS:-}" ]]; then
    TRAIN_ITERS=$(((TRAIN_TOKENS + TOKENS_PER_STEP - 1) / TOKENS_PER_STEP))
elif [[ -n "${TRAIN_ITERS:-}" ]]; then
    TRAIN_ITERS="${TRAIN_ITERS}"
else
    TRAIN_TOKENS="${DEFAULT_TRAIN_TOKENS}"
    TRAIN_ITERS=$(((TRAIN_TOKENS + TOKENS_PER_STEP - 1) / TOKENS_PER_STEP))
fi
TOTAL_TRAIN_TOKENS=$((TRAIN_ITERS * TOKENS_PER_STEP))

LR="${LR:-3e-4}"
MIN_LR="${MIN_LR:-3e-5}"
LR_DECAY_STYLE="${LR_DECAY_STYLE:-cosine}"
LR_DECAY_ITERS="${LR_DECAY_ITERS:-${TRAIN_ITERS}}"
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-$((TRAIN_ITERS / 10))}"
if (( LR_WARMUP_ITERS < 1 )); then
    LR_WARMUP_ITERS=1
fi
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
CLIP_GRAD="${CLIP_GRAD:-1.0}"
SEED="${SEED:-1234}"

LOG_INTERVAL="${LOG_INTERVAL:-1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
EVAL_ITERS="${EVAL_ITERS:-10}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
DRY_RUN="${DRY_RUN:-0}"

FP4_OUTLIER_RATIO="${FP4_OUTLIER_RATIO:-0.001}"
FP4_OUTLIER_SELECTION_METHOD="${FP4_OUTLIER_SELECTION_METHOD:-normal_threshold}"
FP4_WEIGHT_INPUT_RATIO="${FP4_WEIGHT_INPUT_RATIO:-0.003}"
FP4_WEIGHT_ROUNDING_GROUP_SIZE="${FP4_WEIGHT_ROUNDING_GROUP_SIZE:-128}"
FP4_WEIGHT_ROUNDING_ROUNDS_PER_GROUP="${FP4_WEIGHT_ROUNDING_ROUNDS_PER_GROUP:-2}"
FP4_WEIGHT_ROUNDING_SELECTION_TOKENS="${FP4_WEIGHT_ROUNDING_SELECTION_TOKENS:-1024}"
FP4_WEIGHT_ROUNDING_AUDIT_TOKENS="${FP4_WEIGHT_ROUNDING_AUDIT_TOKENS:-512}"
FP4_WEIGHT_ROUNDING_AUDIT_MAX_REGRESSION_FRACTION="${FP4_WEIGHT_ROUNDING_AUDIT_MAX_REGRESSION_FRACTION:-0.5}"
FP4_WEIGHT_ROUNDING_OFFDIAG_SHRINK="${FP4_WEIGHT_ROUNDING_OFFDIAG_SHRINK:-0.5}"
FP4_TE_CONFIG_NVFP4="${FP4_TE_CONFIG_NVFP4:-${REPO_ROOT}/megatron/core/extensions/fp4_config.yaml}"
FP4_TE_CONFIG_CUSTOM="${FP4_TE_CONFIG_CUSTOM:-${REPO_ROOT}/megatron/core/extensions/fp4_outlier_config.yaml}"
FP4_QUANTIZER_FACTORY="${FP4_QUANTIZER_FACTORY:-megatron.core.extensions.fp4_outlier_recipe.nvfp4_outlier_quantizer_factory}"
EXTRA_MEGATRON_ARGS="${EXTRA_MEGATRON_ARGS:-}"
LOAD_CHECKPOINT_DIR="${LOAD_CHECKPOINT_DIR:-}"
LOAD_CHECKPOINT_STEP="${LOAD_CHECKPOINT_STEP:-}"

if [[ "${ALLOW_MOCK_DATA}" != "1" ]]; then
    if [[ ! -f "${DATA_PATH}.bin" || ! -f "${DATA_PATH}.idx" ]]; then
        echo "Missing indexed dataset: ${DATA_PATH}.{bin,idx}" >&2
        echo "Set DATA_PATH to a Megatron indexed prefix or set ALLOW_MOCK_DATA=1 for a smoke test." >&2
        exit 2
    fi
fi

CHECKPOINT_LOAD_ARGS=()
if [[ -n "${LOAD_CHECKPOINT_STEP}" && -z "${LOAD_CHECKPOINT_DIR}" ]]; then
    echo "LOAD_CHECKPOINT_STEP requires LOAD_CHECKPOINT_DIR." >&2
    exit 2
fi
if [[ -n "${LOAD_CHECKPOINT_DIR}" ]]; then
    if [[ ! -f "${LOAD_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" ]]; then
        echo "Missing checkpoint tracker: ${LOAD_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" >&2
        exit 2
    fi
    CHECKPOINT_LOAD_ARGS=(
        --load "${LOAD_CHECKPOINT_DIR}"
        --exit-on-missing-checkpoint
    )
    if [[ -n "${LOAD_CHECKPOINT_STEP}" ]]; then
        checkpoint_step_dir=$(printf '%s/iter_%07d' "${LOAD_CHECKPOINT_DIR}" "${LOAD_CHECKPOINT_STEP}")
        if [[ ! -d "${checkpoint_step_dir}" ]]; then
            echo "Missing requested checkpoint: ${checkpoint_step_dir}" >&2
            exit 2
        fi
        CHECKPOINT_LOAD_ARGS+=(--ckpt-step "${LOAD_CHECKPOINT_STEP}")
    fi
fi

TOKENIZER_ARGS=()
case "${TOKENIZER_TYPE}" in
    GPT2BPETokenizer)
        if [[ -z "${VOCAB_FILE}" || -z "${MERGE_FILE}" ]]; then
            echo "GPT2BPETokenizer requires VOCAB_FILE and MERGE_FILE." >&2
            exit 2
        fi
        TOKENIZER_ARGS=(
            --tokenizer-type GPT2BPETokenizer
            --vocab-file "${VOCAB_FILE}"
            --merge-file "${MERGE_FILE}"
            --vocab-size "${VOCAB_SIZE}"
        )
        ;;
    HuggingFaceTokenizer)
        TOKENIZER_ARGS=(
            --tokenizer-type HuggingFaceTokenizer
            --tokenizer-model "${TOKENIZER_MODEL}"
            --vocab-size "${VOCAB_SIZE}"
        )
        ;;
    *)
        echo "Unsupported TOKENIZER_TYPE=${TOKENIZER_TYPE}. Use HuggingFaceTokenizer or GPT2BPETokenizer." >&2
        exit 2
        ;;
esac

MODEL_ARGS=(
    --use-mcore-models
    --num-layers "${NUM_LAYERS}"
    --hidden-size "${HIDDEN_SIZE}"
    --ffn-hidden-size "${FFN_HIDDEN_SIZE}"
    --num-attention-heads "${NUM_ATTENTION_HEADS}"
    --seq-length "${SEQ_LENGTH}"
    --max-position-embeddings "${MAX_POSITION_EMBEDDINGS}"
    --position-embedding-type learned_absolute
    --attention-dropout 0.1
    --hidden-dropout 0.1
    --normalization LayerNorm
    --norm-epsilon 1e-5
    --openai-gelu
    --attention-backend fused
    --make-vocab-size-divisible-by "${MAKE_VOCAB_SIZE_DIVISIBLE_BY}"
    --init-method-std 0.02
    --seed "${SEED}"
)

TRAINING_ARGS=(
    --micro-batch-size "${MICRO_BATCH_SIZE}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --train-iters "${TRAIN_ITERS}"
    --lr-decay-iters "${LR_DECAY_ITERS}"
    --lr-warmup-iters "${LR_WARMUP_ITERS}"
    --lr "${LR}"
    --min-lr "${MIN_LR}"
    --lr-decay-style "${LR_DECAY_STYLE}"
    --clip-grad "${CLIP_GRAD}"
    --weight-decay "${WEIGHT_DECAY}"
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --cross-entropy-loss-fusion
    --calculate-per-token-loss
    --manual-gc
    --empty-unused-memory-level 1
    --bf16
    --grad-reduce-in-bf16
    --use-distributed-optimizer
    --overlap-grad-reduce
)

PARALLEL_ARGS=(
    --tensor-model-parallel-size "${TP_SIZE}"
)
if (( PP_SIZE > 1 )); then
    PARALLEL_ARGS+=(--pipeline-model-parallel-size "${PP_SIZE}")
fi
if (( TP_SIZE > 1 )); then
    PARALLEL_ARGS+=(--sequence-parallel)
fi

if [[ "${ALLOW_MOCK_DATA}" == "1" && ( ! -f "${DATA_PATH}.bin" || ! -f "${DATA_PATH}.idx" ) ]]; then
    DATA_ARGS=(
        --mock-data
        --tokenizer-type NullTokenizer
        --vocab-size "${VOCAB_SIZE}"
        --data-cache-path "${DATA_CACHE_PATH}"
        --split 99,1,0
        --no-create-attention-mask-in-dataloader
        --no-mmap-bin-files
        --num-workers 1
    )
else
    DATA_ARGS=(
        --data-path "${DATA_PATH}"
        "${TOKENIZER_ARGS[@]}"
        --data-cache-path "${DATA_CACHE_PATH}"
        --split 99,1,0
        --no-create-attention-mask-in-dataloader
        --no-mmap-bin-files
        --num-workers 1
    )
fi

LOGGING_ARGS=(
    --log-interval "${LOG_INTERVAL}"
    --eval-iters "${EVAL_ITERS}"
    --eval-interval "${EVAL_INTERVAL}"
    --save-interval "${SAVE_INTERVAL}"
    --ckpt-format torch_dist
    --distributed-timeout-minutes 60
    --log-throughput
    --log-memory-to-tensorboard
    --log-timers-to-tensorboard
    --log-validation-ppl-to-tensorboard
    --log-world-size-to-tensorboard
    --tensorboard-queue-size 1000
    --tensorboard-log-interval 1
)

DISTRIBUTED_ARGS=(
    --nproc_per_node "${GPUS_PER_NODE}"
    --nnodes "${NUM_NODES}"
    --node_rank "${NODE_RANK}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
)

print_command() {
    printf '%q ' "$@"
    printf '\n'
}

build_dtype_args() {
    local variant=$1
    local -n out_args=$2
    out_args=()
    case "${variant}" in
        bf16)
            ;;
        te_nvfp4|te)
            out_args=(
                --fp4-format e2m1
                --fp4-recipe nvfp4
                --te-precision-config-file "${FP4_TE_CONFIG_NVFP4}"
            )
            ;;
        our|ours|ours0p1|ours_0p1)
            out_args=(
                --fp4-format e2m1
                --fp4-recipe custom
                --te-precision-config-file "${FP4_TE_CONFIG_CUSTOM}"
                --fp4-quantizer-factory "${FP4_QUANTIZER_FACTORY}"
                --fp4-outlier-ratio "${FP4_OUTLIER_RATIO}"
                --fp4-outlier-selection-method "${FP4_OUTLIER_SELECTION_METHOD}"
                --fp4-outlier-enable-fast-fprop
                --no-fp4-outlier-enable-dgrad
                --no-fp4-outlier-enable-wgrad
            )
            ;;
        weight_only|elcr_weight_only)
            out_args=(
                --fp4-format e2m1
                --fp4-recipe custom
                --te-precision-config-file "${FP4_TE_CONFIG_CUSTOM}"
                --fp4-quantizer-factory "${FP4_QUANTIZER_FACTORY}"
                --fp4-outlier-ratio 0.0
                --fp4-outlier-selection-method normal_threshold
                --fp4-outlier-enable-weight-rounding
                --fp4-outlier-weight-rounding-group-size "${FP4_WEIGHT_ROUNDING_GROUP_SIZE}"
                --fp4-outlier-weight-rounding-rounds-per-group "${FP4_WEIGHT_ROUNDING_ROUNDS_PER_GROUP}"
                --fp4-outlier-weight-rounding-selection-tokens "${FP4_WEIGHT_ROUNDING_SELECTION_TOKENS}"
                --fp4-outlier-weight-rounding-offdiag-shrink "${FP4_WEIGHT_ROUNDING_OFFDIAG_SHRINK}"
                --no-fp4-outlier-enable-dgrad
                --no-fp4-outlier-enable-wgrad
            )
            ;;
        weight_only_stratified)
            out_args=(
                --fp4-format e2m1
                --fp4-recipe custom
                --te-precision-config-file "${FP4_TE_CONFIG_CUSTOM}"
                --fp4-quantizer-factory "${FP4_QUANTIZER_FACTORY}"
                --fp4-outlier-ratio 0.0
                --fp4-outlier-selection-method normal_threshold
                --fp4-outlier-enable-weight-rounding
                --fp4-outlier-weight-rounding-stratified-sampling
                --fp4-outlier-weight-rounding-stratified-batch-size "${MICRO_BATCH_SIZE}"
                --fp4-outlier-weight-rounding-group-size "${FP4_WEIGHT_ROUNDING_GROUP_SIZE}"
                --fp4-outlier-weight-rounding-rounds-per-group "${FP4_WEIGHT_ROUNDING_ROUNDS_PER_GROUP}"
                --fp4-outlier-weight-rounding-selection-tokens "${FP4_WEIGHT_ROUNDING_SELECTION_TOKENS}"
                --fp4-outlier-weight-rounding-offdiag-shrink "${FP4_WEIGHT_ROUNDING_OFFDIAG_SHRINK}"
                --no-fp4-outlier-enable-dgrad
                --no-fp4-outlier-enable-wgrad
            )
            ;;
        weight_only_crossfit)
            out_args=(
                --fp4-format e2m1
                --fp4-recipe custom
                --te-precision-config-file "${FP4_TE_CONFIG_CUSTOM}"
                --fp4-quantizer-factory "${FP4_QUANTIZER_FACTORY}"
                --fp4-outlier-ratio 0.0
                --fp4-outlier-selection-method normal_threshold
                --fp4-outlier-enable-weight-rounding
                --fp4-outlier-weight-rounding-stratified-sampling
                --fp4-outlier-weight-rounding-stratified-batch-size "${MICRO_BATCH_SIZE}"
                --fp4-outlier-weight-rounding-crossfit-audit
                --fp4-outlier-weight-rounding-audit-tokens "${FP4_WEIGHT_ROUNDING_AUDIT_TOKENS}"
                --fp4-outlier-weight-rounding-audit-max-regression-fraction "${FP4_WEIGHT_ROUNDING_AUDIT_MAX_REGRESSION_FRACTION}"
                --fp4-outlier-weight-rounding-group-size "${FP4_WEIGHT_ROUNDING_GROUP_SIZE}"
                --fp4-outlier-weight-rounding-rounds-per-group "${FP4_WEIGHT_ROUNDING_ROUNDS_PER_GROUP}"
                --fp4-outlier-weight-rounding-selection-tokens "${FP4_WEIGHT_ROUNDING_SELECTION_TOKENS}"
                --fp4-outlier-weight-rounding-offdiag-shrink "${FP4_WEIGHT_ROUNDING_OFFDIAG_SHRINK}"
                --no-fp4-outlier-enable-dgrad
                --no-fp4-outlier-enable-wgrad
            )
            ;;
        weight_input_r0p003_crossfit)
            out_args=(
                --fp4-format e2m1
                --fp4-recipe custom
                --te-precision-config-file "${FP4_TE_CONFIG_CUSTOM}"
                --fp4-quantizer-factory "${FP4_QUANTIZER_FACTORY}"
                --fp4-outlier-ratio "${FP4_WEIGHT_INPUT_RATIO}"
                --fp4-outlier-selection-method normal_threshold
                --fp4-outlier-enable-fast-fprop
                --fp4-outlier-enable-weight-rounding
                --fp4-outlier-weight-rounding-stratified-sampling
                --fp4-outlier-weight-rounding-stratified-batch-size "${MICRO_BATCH_SIZE}"
                --fp4-outlier-weight-rounding-crossfit-audit
                --fp4-outlier-weight-rounding-audit-tokens "${FP4_WEIGHT_ROUNDING_AUDIT_TOKENS}"
                --fp4-outlier-weight-rounding-audit-max-regression-fraction "${FP4_WEIGHT_ROUNDING_AUDIT_MAX_REGRESSION_FRACTION}"
                --fp4-outlier-weight-rounding-group-size "${FP4_WEIGHT_ROUNDING_GROUP_SIZE}"
                --fp4-outlier-weight-rounding-rounds-per-group "${FP4_WEIGHT_ROUNDING_ROUNDS_PER_GROUP}"
                --fp4-outlier-weight-rounding-selection-tokens "${FP4_WEIGHT_ROUNDING_SELECTION_TOKENS}"
                --fp4-outlier-weight-rounding-offdiag-shrink "${FP4_WEIGHT_ROUNDING_OFFDIAG_SHRINK}"
                --no-fp4-outlier-enable-dgrad
                --no-fp4-outlier-enable-wgrad
            )
            ;;
        weight_only_bcr)
            out_args=(
                --fp4-format e2m1
                --fp4-recipe custom
                --te-precision-config-file "${FP4_TE_CONFIG_CUSTOM}"
                --fp4-quantizer-factory "${FP4_QUANTIZER_FACTORY}"
                --fp4-outlier-ratio 0.0
                --fp4-outlier-selection-method normal_threshold
                --fp4-outlier-enable-weight-rounding
                --fp4-outlier-weight-rounding-dgrad-consistency
                --fp4-outlier-weight-rounding-group-size "${FP4_WEIGHT_ROUNDING_GROUP_SIZE}"
                --fp4-outlier-weight-rounding-rounds-per-group "${FP4_WEIGHT_ROUNDING_ROUNDS_PER_GROUP}"
                --fp4-outlier-weight-rounding-selection-tokens "${FP4_WEIGHT_ROUNDING_SELECTION_TOKENS}"
                --fp4-outlier-weight-rounding-offdiag-shrink "${FP4_WEIGHT_ROUNDING_OFFDIAG_SHRINK}"
                --no-fp4-outlier-enable-dgrad
                --no-fp4-outlier-enable-wgrad
            )
            ;;
        weight_only_microbatch)
            out_args=(
                --fp4-format e2m1
                --fp4-recipe custom
                --te-precision-config-file "${FP4_TE_CONFIG_CUSTOM}"
                --fp4-quantizer-factory "${FP4_QUANTIZER_FACTORY}"
                --fp4-outlier-ratio 0.0
                --fp4-outlier-selection-method normal_threshold
                --fp4-outlier-enable-weight-rounding
                --no-fp4-outlier-weight-rounding-reuse-generation-payload
                --fp4-outlier-weight-rounding-group-size "${FP4_WEIGHT_ROUNDING_GROUP_SIZE}"
                --fp4-outlier-weight-rounding-rounds-per-group "${FP4_WEIGHT_ROUNDING_ROUNDS_PER_GROUP}"
                --fp4-outlier-weight-rounding-selection-tokens "${FP4_WEIGHT_ROUNDING_SELECTION_TOKENS}"
                --fp4-outlier-weight-rounding-offdiag-shrink "${FP4_WEIGHT_ROUNDING_OFFDIAG_SHRINK}"
                --no-fp4-outlier-enable-dgrad
                --no-fp4-outlier-enable-wgrad
            )
            ;;
        weight_input_r0p003|elcr_weight_input_r0p003)
            out_args=(
                --fp4-format e2m1
                --fp4-recipe custom
                --te-precision-config-file "${FP4_TE_CONFIG_CUSTOM}"
                --fp4-quantizer-factory "${FP4_QUANTIZER_FACTORY}"
                --fp4-outlier-ratio "${FP4_WEIGHT_INPUT_RATIO}"
                --fp4-outlier-selection-method normal_threshold
                --fp4-outlier-enable-fast-fprop
                --fp4-outlier-enable-weight-rounding
                --fp4-outlier-weight-rounding-group-size "${FP4_WEIGHT_ROUNDING_GROUP_SIZE}"
                --fp4-outlier-weight-rounding-rounds-per-group "${FP4_WEIGHT_ROUNDING_ROUNDS_PER_GROUP}"
                --fp4-outlier-weight-rounding-selection-tokens "${FP4_WEIGHT_ROUNDING_SELECTION_TOKENS}"
                --fp4-outlier-weight-rounding-offdiag-shrink "${FP4_WEIGHT_ROUNDING_OFFDIAG_SHRINK}"
                --no-fp4-outlier-enable-dgrad
                --no-fp4-outlier-enable-wgrad
            )
            ;;
        weight_input_r0p003_bcr)
            out_args=(
                --fp4-format e2m1
                --fp4-recipe custom
                --te-precision-config-file "${FP4_TE_CONFIG_CUSTOM}"
                --fp4-quantizer-factory "${FP4_QUANTIZER_FACTORY}"
                --fp4-outlier-ratio "${FP4_WEIGHT_INPUT_RATIO}"
                --fp4-outlier-selection-method normal_threshold
                --fp4-outlier-enable-fast-fprop
                --fp4-outlier-enable-weight-rounding
                --fp4-outlier-weight-rounding-dgrad-consistency
                --fp4-outlier-weight-rounding-group-size "${FP4_WEIGHT_ROUNDING_GROUP_SIZE}"
                --fp4-outlier-weight-rounding-rounds-per-group "${FP4_WEIGHT_ROUNDING_ROUNDS_PER_GROUP}"
                --fp4-outlier-weight-rounding-selection-tokens "${FP4_WEIGHT_ROUNDING_SELECTION_TOKENS}"
                --fp4-outlier-weight-rounding-offdiag-shrink "${FP4_WEIGHT_ROUNDING_OFFDIAG_SHRINK}"
                --no-fp4-outlier-enable-dgrad
                --no-fp4-outlier-enable-wgrad
            )
            ;;
        quartet_ii|quartet-ii|qii)
            echo "quartet_ii is not wired in this release; use the official Quartet-II implementation for this control." >&2
            return 2
            ;;
        *)
            echo "Unsupported variant: ${variant}. Expected bf16, te, our, weight_only, weight_only_stratified, weight_only_crossfit, weight_only_bcr, weight_only_microbatch, weight_input_r0p003, weight_input_r0p003_crossfit, or weight_input_r0p003_bcr." >&2
            return 2
            ;;
    esac
}

IFS=',' read -r -a VARIANTS <<< "${RUN_VARIANTS}"
read -r -a EXTRA_ARGS <<< "${EXTRA_MEGATRON_ARGS}"

cat <<EOF
GPT-2 345M OpenWebText plan
  variants: ${RUN_VARIANTS}
  data: ${DATA_PATH}
  gpus: ${GPUS_PER_NODE} / CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
  seq length: ${SEQ_LENGTH}
  global batch size: ${GLOBAL_BATCH_SIZE}
  tokens per step: ${TOKENS_PER_STEP}
  train iters: ${TRAIN_ITERS}
  total train tokens: ${TOTAL_TRAIN_TOKENS}
  save interval: ${SAVE_INTERVAL}
  load checkpoint: ${LOAD_CHECKPOINT_DIR:-none}
  load step: ${LOAD_CHECKPOINT_STEP:-latest}
  wandb project: ${WANDB_PROJECT}
EOF

for raw_variant in "${VARIANTS[@]}"; do
    variant=$(echo "${raw_variant}" | xargs)
    if [[ -z "${variant}" ]]; then
        continue
    fi

    dtype_args=()
    build_dtype_args "${variant}" dtype_args

    run_name="${RUN_NAME_PREFIX}_${variant}_s${SEED}_seq${SEQ_LENGTH}_gb${GLOBAL_BATCH_SIZE}_it${TRAIN_ITERS}"
    variant_ckpt="${CHECKPOINT_DIR}/${variant}"
    variant_tb="${TENSORBOARD_DIR}/${variant}"
    variant_log="${LOG_DIR}/${run_name}.log"
    mkdir -p "${variant_ckpt}" "${variant_tb}"

    cmd=(
        torchrun
        "${DISTRIBUTED_ARGS[@]}"
        pretrain_gpt.py
        "${MODEL_ARGS[@]}"
        "${TRAINING_ARGS[@]}"
        "${dtype_args[@]}"
        "${PARALLEL_ARGS[@]}"
        "${DATA_ARGS[@]}"
        "${LOGGING_ARGS[@]}"
        "${CHECKPOINT_LOAD_ARGS[@]}"
        --save "${variant_ckpt}"
        --tensorboard-dir "${variant_tb}"
        --wandb-exp-name "${run_name}"
        --wandb-project "${WANDB_PROJECT}"
        --wandb-save-dir "${LOG_DIR}/wandb_fp4"
        "${EXTRA_ARGS[@]}"
    )

    echo "Running ${variant}; log: ${variant_log}"
    print_command "${cmd[@]}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        continue
    fi
    "${cmd[@]}" 2>&1 | tee "${variant_log}"
done
