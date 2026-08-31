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

DEFAULT_DATA_PREFIX="${DEFAULT_DATA_PREFIX:-${REPO_ROOT}/datasets/openwebtext_gpt2/bpe_openwebtext}"
RAW_JSONL="${RAW_JSONL:-}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-${REPO_ROOT}/datasets/openwebtext_gpt2/bpe_openwebtext}"
JSON_KEYS="${JSON_KEYS:-text}"
WORKERS="${WORKERS:-32}"
DRY_RUN="${DRY_RUN:-0}"

TOKENIZER_TYPE="${TOKENIZER_TYPE:-HuggingFaceTokenizer}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-gpt2}"
VOCAB_FILE="${VOCAB_FILE:-}"
MERGE_FILE="${MERGE_FILE:-}"

if [[ -f "${DEFAULT_DATA_PREFIX}.bin" && -f "${DEFAULT_DATA_PREFIX}.idx" && -z "${RAW_JSONL}" ]]; then
    echo "Existing GPT-2 BPE OpenWebText indexed dataset is ready:"
    echo "${DEFAULT_DATA_PREFIX}"
    exit 0
fi

if [[ -z "${RAW_JSONL}" ]]; then
    echo "RAW_JSONL is required when the default indexed dataset is not used." >&2
    echo "Expected JSONL rows like: {\"text\": \"...\"}" >&2
    exit 2
fi
if [[ ! -f "${RAW_JSONL}" ]]; then
    echo "RAW_JSONL does not exist: ${RAW_JSONL}" >&2
    exit 2
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
        )
        ;;
    HuggingFaceTokenizer)
        TOKENIZER_ARGS=(
            --tokenizer-type HuggingFaceTokenizer
            --tokenizer-model "${TOKENIZER_MODEL}"
        )
        ;;
    *)
        echo "Unsupported TOKENIZER_TYPE=${TOKENIZER_TYPE}. Use HuggingFaceTokenizer or GPT2BPETokenizer." >&2
        exit 2
        ;;
esac

mkdir -p "$(dirname "${OUTPUT_PREFIX}")"

cmd=(
    python
    tools/preprocess_data.py
    --input "${RAW_JSONL}"
    --output-prefix "${OUTPUT_PREFIX}"
    --json-keys "${JSON_KEYS}"
    "${TOKENIZER_ARGS[@]}"
    --append-eod
    --workers "${WORKERS}"
)

printf '%q ' "${cmd[@]}"
printf '\n'
if [[ "${DRY_RUN}" == "1" ]]; then
    exit 0
fi

"${cmd[@]}"

echo "Megatron data prefix:"
echo "${OUTPUT_PREFIX}_${JSON_KEYS}_document"
