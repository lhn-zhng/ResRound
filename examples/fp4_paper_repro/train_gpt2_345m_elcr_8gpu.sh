#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

# Keep the GPT-2 345M training configuration in the shared entrypoint intact.
# This wrapper only fixes the launch to eight GPUs and selects the two ELCR
# ablations requested for comparison.
export MEGATRON_CONDA_ENV="${MEGATRON_CONDA_ENV:-transformer_engine}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export RUN_VARIANTS="${RUN_VARIANTS:-weight_only,weight_input_r0p003}"

cd "${REPO_ROOT}"
exec bash "${SCRIPT_DIR}/train_gpt2_345m_openwebtext.sh"
