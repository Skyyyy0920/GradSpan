#!/usr/bin/env bash
# Server setup for GradSpan-KD.
# Usage:  CUDA_TAG=cu121 bash setup_server.sh
# CUDA_TAG should match your driver / NVCC version (cu118 / cu121 / cu124 / cu126 ...).
set -euo pipefail

echo "==> GradSpan-KD server setup"
echo "    CUDA_TAG = ${CUDA_TAG:-cu121}  (override with CUDA_TAG=cuXYZ bash setup_server.sh)"

CUDA_TAG="${CUDA_TAG:-cu121}"

# ---- 0. sanity ----
command -v python >/dev/null 2>&1 || { echo "ERR: python not on PATH"; exit 1; }
command -v pip    >/dev/null 2>&1 || { echo "ERR: pip not on PATH";    exit 1; }
python -c "import sys; assert sys.version_info >= (3,10), sys.version" \
    || { echo "ERR: need Python >= 3.10"; exit 1; }

# ---- 1. CUDA torch ----
echo
echo "==> [1/4] Installing PyTorch (CUDA wheel: ${CUDA_TAG})"
pip install --upgrade pip
pip install torch --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

# ---- 2. Other Python deps ----
echo
echo "==> [2/4] Installing other Python dependencies"
pip install -r requirements.txt

# ---- 3. Verify CUDA ----
echo
echo "==> [3/4] Verifying CUDA"
python - <<'PY'
import torch
assert torch.cuda.is_available(), \
    "CUDA not available — check driver matches CUDA_TAG, or `nvidia-smi`"
print(f"  torch        = {torch.__version__}")
print(f"  cuda runtime = {torch.version.cuda}")
print(f"  device count = {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"    [{i}] {props.name}  ({props.total_memory / 1024**3:.1f} GiB)")
PY

# ---- 3.5. Shared-GPU snapshot ----
echo
echo "==> Shared-GPU snapshot (this server is multi-tenant):"
nvidia-smi --query-gpu=index,name,memory.free,memory.used,utilization.gpu --format=csv
echo
echo "  -> Before launching any experiment, pick a GPU whose memory.free is above"
echo "     the per-job threshold (20 GB for E0/small-E1, 40 GB for E2+), then:"
echo "        export CUDA_VISIBLE_DEVICES=<idx>"
echo "        nvidia-smi --query-gpu=gpu_uuid -i <idx> --format=csv,noheader    # log this"
echo "     See CLAUDE.md > 'Shared-GPU coordination' for the full pattern,"
echo "     including the >60-min wait fallback and the OOM-recovery rules."

# ---- 4. Optional: Codex MCP ----
echo
echo "==> [4/4] (optional) Wiring Codex MCP for /research-review and /novelty-check cross-checks"
if command -v claude >/dev/null 2>&1; then
    if command -v codex >/dev/null 2>&1; then
        claude mcp add codex -s user -- codex mcp-server || true
        echo "    Codex MCP wired."
    else
        echo "    'codex' CLI not found; skipping. To enable later:"
        echo "        npm i -g @anthropic-ai/codex"
        echo "        claude mcp add codex -s user -- codex mcp-server"
    fi
else
    echo "    Claude Code CLI ('claude') not found; install with:"
    echo "        npm i -g @anthropic-ai/claude-code"
fi

# ---- 5. Smoke test: re-run the CPU pilot ----
echo
echo "==> Smoke test: re-running pilot/pilot_gradgeom.py (~1 min)"
python pilot/pilot_gradgeom.py | tail -25

echo
echo "============================================================"
echo "==> Setup complete."
echo "    Next: open Claude Code in this directory with \`claude\`."
echo "    CLAUDE.md will auto-load. Follow §Next action - one-click resume."
echo "============================================================"
