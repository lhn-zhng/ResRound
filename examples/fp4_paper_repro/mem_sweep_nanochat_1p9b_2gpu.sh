#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
LAUNCHER="${SCRIPT_DIR}/train_nanochat_1p9b_fineweb_edu.sh"

if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
    source /opt/conda/etc/profile.d/conda.sh
    if [[ "${CONDA_DEFAULT_ENV:-}" != "${MEGATRON_CONDA_ENV:-transformer_engine}" ]]; then
        conda activate "${MEGATRON_CONDA_ENV:-transformer_engine}"
    fi
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
TP_SIZE="${TP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"
CP_SIZE="${CP_SIZE:-1}"
RUN_VARIANTS="${RUN_VARIANTS:-bf16,te,our}"
MBS_LIST="${MBS_LIST:-8 16 24 32 40 48 56 64}"
TRAIN_ITERS="${TRAIN_ITERS:-3}"
SEQ_LENGTH="${SEQ_LENGTH:-1024}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-2}"
STOP_VARIANT_ON_FAIL="${STOP_VARIANT_ON_FAIL:-1}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/nanochat_1p9b_mem_sweep_2gpu}"
mkdir -p "${LOG_ROOT}"

visible_count=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
if (( visible_count < GPUS_PER_NODE )); then
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} exposes ${visible_count} device(s), but GPUS_PER_NODE=${GPUS_PER_NODE}." >&2
    exit 2
fi

detected_count=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
if [[ "${detected_count}" =~ ^[0-9]+$ ]] && (( detected_count < GPUS_PER_NODE )); then
    echo "nvidia-smi sees only ${detected_count} GPU(s), but GPUS_PER_NODE=${GPUS_PER_NODE}." >&2
    echo "Run this script in the shell where nvidia-smi shows the two free RTX PRO 6000 GPUs." >&2
    exit 2
fi

timestamp=$(date +%Y%m%d_%H%M%S)
SUMMARY="${LOG_ROOT}/summary_${timestamp}.tsv"
printf 'variant\tmicro_batch\tglobal_batch\trc\tstatus\tduration_s\tmax_mem_mib_per_gpu\tlog\n' > "${SUMMARY}"

child_pgid=""
cleanup() {
    if [[ -n "${child_pgid}" ]] && kill -0 "${child_pgid}" 2>/dev/null; then
        kill -TERM "-${child_pgid}" 2>/dev/null || true
    fi
}
trap cleanup INT TERM

sample_gpu_mem() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null |
        head -n "${GPUS_PER_NODE}" |
        awk '{gsub(/[^0-9]/, "", $1); print int($1)}'
}

join_by_comma() {
    local IFS=,
    echo "$*"
}

case_status() {
    local rc=$1
    local log=$2
    if (( rc == 0 )); then
        echo ok
    elif rg -qi 'out of memory|cuda error: out of memory|CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED' "${log}"; then
        echo oom
    else
        echo fail
    fi
}

print_key_log_lines() {
    local log=$1
    rg -i 'iteration|elapsed time|throughput|tokens/sec|lm loss|loss scale|out of memory|cuda error|traceback|error|matched to quant config|FP4 outlier custom|memory' "${log}" |
        tail -n 80 || tail -n 40 "${log}" || true
}

run_case() {
    local variant=$1
    local mbs=$2
    local gbs=$((mbs * GPUS_PER_NODE))
    local log="${LOG_ROOT}/${variant}_mbs${mbs}_gbs${gbs}_${timestamp}.log"
    local port=$((20000 + RANDOM % 20000))
    local -a max_mem=()
    local i mem rc status start_ts end_ts duration

    for ((i = 0; i < GPUS_PER_NODE; i++)); do
        max_mem[i]=0
    done

    echo "=== variant=${variant} micro_batch=${mbs} global_batch=${gbs} log=${log} ==="
    start_ts=$(date +%s)

    setsid env \
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
        WANDB_MODE=disabled \
        SAVE_CHECKPOINT=0 \
        EVAL_ITERS=1 \
        EVAL_INTERVAL=1000000 \
        ALLOW_MOCK_DATA=1 \
        REQUIRE_FINEWEB_COMPLETE=0 \
        RUN_VARIANTS="${variant}" \
        GPUS_PER_NODE="${GPUS_PER_NODE}" \
        TP_SIZE="${TP_SIZE}" \
        PP_SIZE="${PP_SIZE}" \
        CP_SIZE="${CP_SIZE}" \
        MICRO_BATCH_SIZE="${mbs}" \
        GLOBAL_BATCH_SIZE="${gbs}" \
        TRAIN_ITERS="${TRAIN_ITERS}" \
        SEQ_LENGTH="${SEQ_LENGTH}" \
        LOG_INTERVAL=1 \
        MASTER_PORT="${port}" \
        bash "${LAUNCHER}" > "${log}" 2>&1 &

    child_pgid=$!

    while kill -0 "${child_pgid}" 2>/dev/null; do
        i=0
        while read -r mem; do
            if [[ -n "${mem}" ]] && (( mem > max_mem[i] )); then
                max_mem[i]=${mem}
            fi
            i=$((i + 1))
        done < <(sample_gpu_mem)
        sleep "${SAMPLE_INTERVAL}"
    done

    set +e
    wait "${child_pgid}"
    rc=$?
    set -e
    child_pgid=""

    end_ts=$(date +%s)
    duration=$((end_ts - start_ts))
    status=$(case_status "${rc}" "${log}")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${variant}" "${mbs}" "${gbs}" "${rc}" "${status}" "${duration}" "$(join_by_comma "${max_mem[@]}")" "${log}" |
        tee -a "${SUMMARY}"

    print_key_log_lines "${log}"
    echo

    [[ "${status}" == "ok" ]]
}

echo "Nanochat 1.9B 2-GPU memory sweep"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  variants=${RUN_VARIANTS}"
echo "  micro batches=${MBS_LIST}"
echo "  train iters=${TRAIN_ITERS}, seq=${SEQ_LENGTH}, tp/pp/cp=${TP_SIZE}/${PP_SIZE}/${CP_SIZE}"
echo "  summary=${SUMMARY}"

IFS=',' read -r -a variants <<< "${RUN_VARIANTS}"
for raw_variant in "${variants[@]}"; do
    variant=$(xargs <<< "${raw_variant}")
    [[ -z "${variant}" ]] && continue
    for mbs in ${MBS_LIST}; do
        if ! run_case "${variant}" "${mbs}"; then
            if [[ "${STOP_VARIANT_ON_FAIL}" == "1" ]]; then
                echo "Stop ${variant} sweep after first non-ok case."
                break
            fi
        fi
    done
done

echo "Summary: ${SUMMARY}"
