#!/usr/bin/env bash

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

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0a}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
# Keep scalar metrics online but avoid final-run stalls uploading large captured console logs.
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"

RUN_NAME="${RUN_NAME:-quartet_c4_100m_fprop_input_6k_ours0p1_fast}"
WANDB_PROJECT="${WANDB_PROJECT:-megatron_fp4_paper_repro_train_quartet_c4_100m}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${REPO_ROOT}/tensorboard_logs/fp4_paper_matrix/${RUN_NAME}}"
DATA_CACHE_PATH="${DATA_CACHE_PATH:-${REPO_ROOT}/benchmark_cache_fp4_paper_matrix}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/datasets/enwiki_openwebtext_llama3/merge}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-${REPO_ROOT}/tokenizers/Llama-3.1-8B-Instruct}"
MASTER_PORT="${MASTER_PORT:-6221}"
TRAIN_ITERS="${TRAIN_ITERS:-6000}"
LR_DECAY_ITERS="${LR_DECAY_ITERS:-6000}"
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-600}"

FP4_FAST_FPROP_ARGS=()
case "${FP4_OUTLIER_ENABLE_FAST_FPROP:-1}" in
    1|true|TRUE|yes|YES|on|ON)
        FP4_FAST_FPROP_ARGS+=(--fp4-outlier-enable-fast-fprop)
        ;;
    0|false|FALSE|no|NO|off|OFF)
        ;;
    *)
        echo "Invalid FP4_OUTLIER_ENABLE_FAST_FPROP=${FP4_OUTLIER_ENABLE_FAST_FPROP}; expected 0/1 or true/false" >&2
        exit 2
        ;;
esac

mkdir -p "${LOG_DIR}" "${TENSORBOARD_DIR}" "${DATA_CACHE_PATH}" "${LOG_DIR}/wandb_fp4"

torchrun \
    --nproc_per_node 1 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port "${MASTER_PORT}" \
    pretrain_gpt.py \
    --use-mcore-models \
    --num-layers 8 \
    --hidden-size 1024 \
    --ffn-hidden-size 2816 \
    --num-attention-heads 8 \
    --kv-channels 128 \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --position-embedding-type rope \
    --rotary-percent 1.0 \
    --rotary-base 10000 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --swiglu \
    --normalization RMSNorm \
    --norm-epsilon 1e-5 \
    --disable-bias-linear \
    --untie-embeddings-and-output-weights \
    --attention-backend fused \
    --apply-layernorm-1p \
    --make-vocab-size-divisible-by 128 \
    --init-method-std 0.0134 \
    --seed 1234 \
    --micro-batch-size 64 \
    --global-batch-size 512 \
    --train-iters "${TRAIN_ITERS}" \
    --lr-decay-iters "${LR_DECAY_ITERS}" \
    --lr-warmup-iters "${LR_WARMUP_ITERS}" \
    --lr 9e-4 \
    --min-lr 9e-5 \
    --lr-decay-style cosine \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --cross-entropy-loss-fusion \
    --calculate-per-token-loss \
    --manual-gc \
    --empty-unused-memory-level 1 \
    --bf16 \
    --grad-reduce-in-bf16 \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --fp4-format e2m1 \
    --fp4-recipe custom \
    --te-precision-config-file "${REPO_ROOT}/megatron/core/extensions/fp4_outlier_config.yaml" \
    --fp4-quantizer-factory megatron.core.extensions.fp4_outlier_recipe.nvfp4_outlier_quantizer_factory \
    --fp4-outlier-ratio 0.001 \
    --fp4-outlier-selection-method normal_threshold \
    "${FP4_FAST_FPROP_ARGS[@]}" \
    --no-fp4-outlier-enable-dgrad \
    --no-fp4-outlier-enable-wgrad \
    --tensor-model-parallel-size 1 \
    --data-path "${DATA_PATH}" \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model "${TOKENIZER_MODEL}" \
    --vocab-size 32000 \
    --data-cache-path "${DATA_CACHE_PATH}" \
    --split 99,1,0 \
    --no-create-attention-mask-in-dataloader \
    --no-mmap-bin-files \
    --num-workers 1 \
    --log-interval 1 \
    --eval-iters 10 \
    --eval-interval 100 \
    --save-interval 100000 \
    --ckpt-format torch_dist \
    --distributed-timeout-minutes 60 \
    --log-throughput \
    --tensorboard-dir "${TENSORBOARD_DIR}" \
    --log-memory-to-tensorboard \
    --log-timers-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --log-world-size-to-tensorboard \
    --tensorboard-queue-size 1000 \
    --tensorboard-log-interval 1 \
    --wandb-exp-name "${RUN_NAME}-gpu1-tp1-cp1-seq512" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-save-dir "${LOG_DIR}/wandb_fp4"
