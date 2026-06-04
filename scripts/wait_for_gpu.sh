#!/usr/bin/env bash
# Wait for a free GPU on a shared server (e.g. multi-tenant A6000 box).
# Prints the index of the first GPU with at least MIN_FREE_GB free memory.
#
# Usage:
#   GPU=$(./scripts/wait_for_gpu.sh)          # defaults: 40 GB free, poll 30s, no timeout
#   GPU=$(./scripts/wait_for_gpu.sh 20 15)    # need >=20 GB, poll every 15s
#   MAX_WAIT_MIN=120 GPU=$(./scripts/wait_for_gpu.sh 40)   # give up after 2h
#
# Then pin your job:
#   export CUDA_VISIBLE_DEVICES=$GPU
#   python experiments/...
#
# Notes:
#  - A6000 has ~48 GB. Default MIN_FREE_GB=40 leaves 8 GB headroom for other users.
#  - Drop to 20 GB for small experiments (E0 spectrum, E1 student-only).
#  - Exits 0 with the GPU index on stdout; exits 2 on timeout (with message to stderr).
set -euo pipefail

MIN_FREE_GB="${1:-40}"
POLL_SEC="${2:-30}"
MAX_WAIT_MIN="${MAX_WAIT_MIN:-0}"   # 0 = wait forever

MIN_FREE_MB=$((MIN_FREE_GB * 1024))
start_ts=$(date +%s)

command -v nvidia-smi >/dev/null 2>&1 \
    || { echo "ERR: nvidia-smi not on PATH" >&2; exit 1; }

while true; do
    free_gpu="$(
        nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
            | awk -v t="${MIN_FREE_MB}" -F, '{gsub(/ /,""); if ($2 >= t) { print $1; exit }}'
    )"

    if [[ -n "${free_gpu}" ]]; then
        echo "${free_gpu}"
        exit 0
    fi

    now=$(date +%s)
    elapsed_min=$(( (now - start_ts) / 60 ))
    if (( MAX_WAIT_MIN > 0 && elapsed_min >= MAX_WAIT_MIN )); then
        echo "ERR: no GPU with ${MIN_FREE_GB} GB free after ${MAX_WAIT_MIN} min" >&2
        nvidia-smi --query-gpu=index,memory.free,memory.used --format=csv >&2
        exit 2
    fi

    echo "[wait_for_gpu] no GPU with ${MIN_FREE_GB} GB free; sleeping ${POLL_SEC}s (elapsed ${elapsed_min}m)" >&2
    sleep "${POLL_SEC}"
done
