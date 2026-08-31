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
    export CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES=$(make_cuda_visible_devices "${DETECTED_GPUS}" 8)
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

MODEL_NAME="${MODEL_NAME:-nanochat_1p9b_fineweb_edu}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-${MODEL_NAME}}"
RUN_VARIANTS="${RUN_VARIANTS:-bf16}"
WANDB_PROJECT="${WANDB_PROJECT:-megatron_fp4_paper_repro_train_nanochat_1p9b}"
PLAN_LABEL="${PLAN_LABEL:-Nanochat 1.9B FineWeb-Edu}"

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/checkpoints/fp4_paper_repro/${MODEL_NAME}}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${REPO_ROOT}/tensorboard_logs/fp4_paper_repro/${MODEL_NAME}}"
DATA_CACHE_PATH="${DATA_CACHE_PATH:-${REPO_ROOT}/benchmark_cache_fp4_paper_repro/${MODEL_NAME}}"
mkdir -p "${LOG_DIR}" "${CHECKPOINT_DIR}" "${TENSORBOARD_DIR}" "${DATA_CACHE_PATH}" "${LOG_DIR}/wandb_fp4"

FINEWEB_ROOT="${FINEWEB_ROOT:-${REPO_ROOT}/datasets/fineweb_edu_llama3_38b}"
DATA_PATH_ARGS_FILE="${DATA_PATH_ARGS_FILE:-${FINEWEB_ROOT}/shards/data_path_args.txt}"
FINEWEB_MANIFEST="${FINEWEB_MANIFEST:-${FINEWEB_ROOT}/manifest.json}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/share/models/Llama-3.1-8B-Instruct}"
ALLOW_MOCK_DATA="${ALLOW_MOCK_DATA:-0}"
REQUIRE_FINEWEB_COMPLETE="${REQUIRE_FINEWEB_COMPLETE:-0}"

GPUS_PER_NODE="${GPUS_PER_NODE:-${DEFAULT_GPUS_PER_NODE}}"
NUM_NODES="${NUM_NODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-6241}"
TP_SIZE="${TP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"
CP_SIZE="${CP_SIZE:-1}"
ENABLE_SEQUENCE_PARALLEL="${ENABLE_SEQUENCE_PARALLEL:-$((TP_SIZE > 1 ? 1 : 0))}"

# Nanochat-like 1.9B shape from the old local reproduction script.
NUM_LAYERS="${NUM_LAYERS:-32}"
HIDDEN_SIZE="${HIDDEN_SIZE:-2048}"
FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-8192}"
NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-16}"
NUM_QUERY_GROUPS="${NUM_QUERY_GROUPS:-16}"
KV_CHANNELS="${KV_CHANNELS:-128}"
USE_GQA="${USE_GQA:-0}"
USE_SWIGLU="${USE_SWIGLU:-0}"
USE_SQUARED_RELU="${USE_SQUARED_RELU:-1}"
QK_LAYERNORM="${QK_LAYERNORM:-1}"
SEQ_LENGTH="${SEQ_LENGTH:-1024}"
MAX_POSITION_EMBEDDINGS="${MAX_POSITION_EMBEDDINGS:-2048}"
ROTARY_BASE="${ROTARY_BASE:-10000}"
ROTARY_PERCENT="${ROTARY_PERCENT:-1.0}"
NORMALIZATION="${NORMALIZATION:-RMSNorm}"
NORM_EPSILON="${NORM_EPSILON:-1e-5}"
VOCAB_SIZE="${VOCAB_SIZE:-128256}"
MAKE_VOCAB_SIZE_DIVISIBLE_BY="${MAKE_VOCAB_SIZE_DIVISIBLE_BY:-128}"

MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
DEFAULT_TRAIN_TOKENS="${DEFAULT_TRAIN_TOKENS:-38000000000}"
TOKENS_PER_STEP=$((GLOBAL_BATCH_SIZE * SEQ_LENGTH))
if [[ -n "${TRAIN_TOKENS:-}" ]]; then
    TRAIN_ITERS=$(((TRAIN_TOKENS + TOKENS_PER_STEP - 1) / TOKENS_PER_STEP))
elif [[ -n "${TRAIN_ITERS:-}" ]]; then
    TRAIN_ITERS="${TRAIN_ITERS}"
    TRAIN_TOKENS=$((TRAIN_ITERS * TOKENS_PER_STEP))
else
    TRAIN_TOKENS="${DEFAULT_TRAIN_TOKENS}"
    TRAIN_ITERS=$(((TRAIN_TOKENS + TOKENS_PER_STEP - 1) / TOKENS_PER_STEP))
fi
TOTAL_TRAIN_TOKENS=$((TRAIN_ITERS * TOKENS_PER_STEP))

LR="${LR:-3e-4}"
MIN_LR="${MIN_LR:-3e-5}"
LR_DECAY_STYLE="${LR_DECAY_STYLE:-WSD}"
LR_DECAY_ITERS="${LR_DECAY_ITERS:-${TRAIN_ITERS}}"
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-$((TRAIN_ITERS / 100))}"
LR_WSD_DECAY_ITERS="${LR_WSD_DECAY_ITERS:-$((TRAIN_ITERS / 10))}"
if (( LR_WARMUP_ITERS < 1 )); then
    LR_WARMUP_ITERS=1
fi
if (( LR_WSD_DECAY_ITERS < 1 )); then
    LR_WSD_DECAY_ITERS=1
fi
OPTIMIZER="${OPTIMIZER:-muon}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.95}"
ADAM_EPS="${ADAM_EPS:-1e-8}"
MUON_MOMENTUM="${MUON_MOMENTUM:-0.95}"
MUON_SCALE_MODE="${MUON_SCALE_MODE:-spectral}"
MUON_NUM_NS_STEPS="${MUON_NUM_NS_STEPS:-5}"
MUON_FP32_MATMUL_PREC="${MUON_FP32_MATMUL_PREC:-medium}"
MUON_TP_MODE="${MUON_TP_MODE:-blockwise}"
MUON_EXTRA_SCALE_FACTOR="${MUON_EXTRA_SCALE_FACTOR:-1.0}"
MUON_USE_NESTEROV="${MUON_USE_NESTEROV:-1}"
MUON_SPLIT_QKV="${MUON_SPLIT_QKV:-1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
CLIP_GRAD="${CLIP_GRAD:-1.0}"
SEED="${SEED:-1234}"

LOG_INTERVAL="${LOG_INTERVAL:-1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
EVAL_ITERS="${EVAL_ITERS:-10}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-${TRAIN_ITERS}}"
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

read_fineweb_tokens() {
    python - "$FINEWEB_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists():
    print("missing 0")
    raise SystemExit
data = json.loads(path.read_text())
print(data.get("status", "unknown"), int(data.get("completed_tokens", 0)))
PY
}

DATA_PATH_PREFIXES=()
USING_FINEWEB_SHARDS=0
if [[ "${ALLOW_MOCK_DATA}" == "1" ]]; then
    :
elif [[ -n "${DATA_PATHS:-}" ]]; then
    read -r -a DATA_PATH_PREFIXES <<< "${DATA_PATHS}"
elif [[ -n "${DATA_PATH:-}" ]]; then
    DATA_PATH_PREFIXES=("${DATA_PATH}")
else
    USING_FINEWEB_SHARDS=1
    if [[ -f "${DATA_PATH_ARGS_FILE}" ]]; then
        read -r -a DATA_PATH_PREFIXES < "${DATA_PATH_ARGS_FILE}"
    fi
fi

if [[ "${ALLOW_MOCK_DATA}" != "1" ]]; then
    if (( USING_FINEWEB_SHARDS == 1 && REQUIRE_FINEWEB_COMPLETE == 1 && DRY_RUN != 1 )); then
        read -r fineweb_status fineweb_tokens <<< "$(read_fineweb_tokens)"
        if [[ "${fineweb_status}" != "complete" || "${fineweb_tokens}" -lt "${TRAIN_TOKENS}" ]]; then
            echo "FineWeb-Edu dataset is not complete enough for this run." >&2
            echo "  manifest: ${FINEWEB_MANIFEST}" >&2
            echo "  status/tokens: ${fineweb_status}/${fineweb_tokens}" >&2
            echo "  required tokens: ${TRAIN_TOKENS}" >&2
            echo "Set REQUIRE_FINEWEB_COMPLETE=0 only for explicit partial-data tests." >&2
            exit 2
        fi
    fi
    if (( ${#DATA_PATH_PREFIXES[@]} == 0 )); then
        echo "No data prefixes found." >&2
        echo "Expected FineWeb-Edu shards in ${DATA_PATH_ARGS_FILE}, or set DATA_PATH/DATA_PATHS." >&2
        echo "For a command smoke test, set ALLOW_MOCK_DATA=1." >&2
        exit 2
    fi
    for prefix in "${DATA_PATH_PREFIXES[@]}"; do
        if [[ ! -f "${prefix}.bin" || ! -f "${prefix}.idx" ]]; then
            echo "Missing indexed dataset prefix: ${prefix}.{bin,idx}" >&2
            exit 2
        fi
    done
fi

if [[ ! -e "${TOKENIZER_MODEL}" && "${ALLOW_MOCK_DATA}" != "1" ]]; then
    echo "Tokenizer path does not exist: ${TOKENIZER_MODEL}" >&2
    exit 2
fi

MODEL_ARGS=(
    --use-mcore-models
    --num-layers "${NUM_LAYERS}"
    --hidden-size "${HIDDEN_SIZE}"
    --ffn-hidden-size "${FFN_HIDDEN_SIZE}"
    --num-attention-heads "${NUM_ATTENTION_HEADS}"
    --kv-channels "${KV_CHANNELS}"
    --seq-length "${SEQ_LENGTH}"
    --max-position-embeddings "${MAX_POSITION_EMBEDDINGS}"
    --position-embedding-type rope
    --rotary-base "${ROTARY_BASE}"
    --rotary-percent "${ROTARY_PERCENT}"
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization "${NORMALIZATION}"
    --norm-epsilon "${NORM_EPSILON}"
    --attention-backend fused
    --make-vocab-size-divisible-by "${MAKE_VOCAB_SIZE_DIVISIBLE_BY}"
    --seed "${SEED}"
    --disable-bias-linear
    --untie-embeddings-and-output-weights
)
if [[ "${USE_GQA}" == "1" ]]; then
    MODEL_ARGS+=(--group-query-attention --num-query-groups "${NUM_QUERY_GROUPS}")
fi
if [[ "${USE_SWIGLU}" == "1" && "${USE_SQUARED_RELU}" == "1" ]]; then
    echo "USE_SWIGLU=1 and USE_SQUARED_RELU=1 are mutually exclusive." >&2
    exit 2
fi
if [[ "${USE_SWIGLU}" == "1" ]]; then
    MODEL_ARGS+=(--swiglu)
fi
if [[ "${USE_SQUARED_RELU}" == "1" ]]; then
    MODEL_ARGS+=(--squared-relu)
fi
if [[ "${QK_LAYERNORM}" == "1" ]]; then
    MODEL_ARGS+=(--qk-layernorm)
fi

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
    --optimizer "${OPTIMIZER}"
    --cross-entropy-loss-fusion
    --calculate-per-token-loss
    --manual-gc
    --empty-unused-memory-level 1
    --bf16
    --grad-reduce-in-bf16
)
if [[ "${LR_DECAY_STYLE}" == "WSD" ]]; then
    TRAINING_ARGS+=(--lr-wsd-decay-iters "${LR_WSD_DECAY_ITERS}" --lr-wsd-decay-style exponential)
fi
case "${OPTIMIZER}" in
    adam|dist_muon|muon)
        TRAINING_ARGS+=(--adam-beta1 "${ADAM_BETA1}" --adam-beta2 "${ADAM_BETA2}" --adam-eps "${ADAM_EPS}")
        ;;
esac
if [[ "${OPTIMIZER}" == "muon" || "${OPTIMIZER}" == "dist_muon" ]]; then
    TRAINING_ARGS+=(
        --muon-momentum "${MUON_MOMENTUM}"
        --muon-scale-mode "${MUON_SCALE_MODE}"
        --muon-num-ns-steps "${MUON_NUM_NS_STEPS}"
        --muon-fp32-matmul-prec "${MUON_FP32_MATMUL_PREC}"
        --muon-tp-mode "${MUON_TP_MODE}"
        --muon-extra-scale-factor "${MUON_EXTRA_SCALE_FACTOR}"
    )
    if [[ "${MUON_USE_NESTEROV}" == "1" ]]; then
        TRAINING_ARGS+=(--muon-use-nesterov)
    fi
    if [[ "${MUON_SPLIT_QKV}" != "1" ]]; then
        TRAINING_ARGS+=(--muon-no-split-qkv)
    fi
fi
if [[ "${OPTIMIZER}" != "muon" ]]; then
    TRAINING_ARGS+=(--use-distributed-optimizer --overlap-grad-reduce)
    if (( GPUS_PER_NODE > 1 || NUM_NODES > 1 || TP_SIZE > 1 || PP_SIZE > 1 )); then
        TRAINING_ARGS+=(--overlap-param-gather)
    fi
fi

PARALLEL_ARGS=(--tensor-model-parallel-size "${TP_SIZE}")
if (( PP_SIZE > 1 )); then
    PARALLEL_ARGS+=(--pipeline-model-parallel-size "${PP_SIZE}")
fi
if (( CP_SIZE > 1 )); then
    PARALLEL_ARGS+=(--context-parallel-size "${CP_SIZE}")
fi
if [[ "${ENABLE_SEQUENCE_PARALLEL}" == "1" ]]; then
    PARALLEL_ARGS+=(--sequence-parallel)
fi

if [[ "${ALLOW_MOCK_DATA}" == "1" ]]; then
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
        --data-path "${DATA_PATH_PREFIXES[@]}"
        --tokenizer-type HuggingFaceTokenizer
        --tokenizer-model "${TOKENIZER_MODEL}"
        --vocab-size "${VOCAB_SIZE}"
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
        te|te_nvfp4|nvfp4)
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
        *)
            echo "Unsupported variant: ${variant}. Expected bf16, te, our, weight_only, weight_input_r0p003, weight_only_crossfit, or weight_input_r0p003_crossfit." >&2
            return 2
            ;;
    esac
}

IFS=',' read -r -a VARIANTS <<< "${RUN_VARIANTS}"
read -r -a EXTRA_ARGS <<< "${EXTRA_MEGATRON_ARGS}"

cat <<EOF
${PLAN_LABEL} plan
  variants: ${RUN_VARIANTS}
  data prefixes: ${#DATA_PATH_PREFIXES[@]}
  tokenizer: ${TOKENIZER_MODEL}
  gpus: ${GPUS_PER_NODE} / CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
  tp/pp/cp: ${TP_SIZE}/${PP_SIZE}/${CP_SIZE}
  seq length: ${SEQ_LENGTH}
  global batch size: ${GLOBAL_BATCH_SIZE}
  tokens per step: ${TOKENS_PER_STEP}
  train iters: ${TRAIN_ITERS}
  total train tokens: ${TOTAL_TRAIN_TOKENS}
  optimizer: ${OPTIMIZER}
  save checkpoint: ${SAVE_CHECKPOINT}, save interval: ${SAVE_INTERVAL}
  wandb project: ${WANDB_PROJECT}
EOF

for raw_variant in "${VARIANTS[@]}"; do
    variant=$(echo "${raw_variant}" | xargs)
    [[ -z "${variant}" ]] && continue

    dtype_args=()
    build_dtype_args "${variant}" dtype_args

    run_name="${RUN_NAME_PREFIX}_${variant}_s${SEED}_seq${SEQ_LENGTH}_gb${GLOBAL_BATCH_SIZE}_tok${TRAIN_TOKENS}"
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
        --tensorboard-dir "${variant_tb}"
        --wandb-exp-name "${run_name}"
        --wandb-project "${WANDB_PROJECT}"
        --wandb-save-dir "${LOG_DIR}/wandb_fp4"
        "${EXTRA_ARGS[@]}"
    )
    if [[ "${SAVE_CHECKPOINT}" == "1" ]]; then
        cmd+=(--save "${variant_ckpt}")
    fi

    echo "Running ${variant}; log: ${variant_log}"
    print_command "${cmd[@]}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        continue
    fi
    "${cmd[@]}" 2>&1 | tee "${variant_log}"
done
