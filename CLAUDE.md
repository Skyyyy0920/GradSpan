# GradSpan-KD — Project Context for Claude Code

> **For Claude Code**: this file is auto-loaded at session start. Read it first, then
> [`IDEA_REPORT.md`](IDEA_REPORT.md) and [`refine-logs/PIPELINE_SUMMARY.md`](refine-logs/PIPELINE_SUMMARY.md)
> for full context. The **"Next action — one-click resume"** section below is the canonical way
> to continue this project autonomously.

## What this is

**GradSpan-KD**: data selection for knowledge distillation via the geometry of the student's
per-sample KD-loss gradient. For each example, take `∇_θ KL(teacher ‖ student)`, Count-Sketch–
project, PCA the per-sample gradient matrix, then keep a D-optimal coreset that spans the
principal gradient subspace; distill only on that coreset.

The project is positioned as a **finding paper**, not a new-algorithm paper:
*for KD, geometric data selection must be done in **parameter-gradient space** — feature-space
(GRAFT-style), saliency-space, and loss-magnitude selection measurably underperform.*

## Current state (frozen snapshot — do not redo)

- **Phases 1–4.5 of `/idea-discovery` are complete.** Literature surveyed (~30 papers), 12 ideas
  brainstormed, 3 piloted on CPU, novelty checked (5–6/10 — recombination of LESS, GRAFT, TAGCOS,
  CCS, ICLR-2025 medium-difficulty KD), critical review done (mock 6/10, borderline accept as a
  finding paper). All deliverables are on disk.
- **CPU pilot evidence** (sklearn-digits MLP KD, ~90s total):
  - KD per-sample gradient matrix is strongly low-rank: effective rank ≈ **23 of 1200** ambient.
  - D-optimal subspace-spanning coreset beats random by **+8.8 acc pts with ~8× lower seed
    variance** at budgets ≈ 1–2× the intrinsic gradient rank (aggressive compression). The
    advantage vanishes by ~4× rank, slightly reverses beyond ~8× rank.
  - Parameter-gradient space has **~7.6× lower normalized rank** than input-saliency space.
  - **Two sub-hypotheses were falsified**: "KD more low-rank than CE" (wash); "structural ≠ hard"
    (correlation is actually +0.59). Both have been dropped/reframed in `IDEA_REPORT.md`.
- **Codex MCP (gpt-5.4) was unavailable** in the original (Windows/CPU) environment, so the
  brainstorm / novelty / review were done in-session by Claude. If Codex is installed on the
  server, re-running `/novelty-check` and `/research-review` for an independent cross-model
  check is recommended before committing all GPU budget.

## Canonical state files (read in this order on every fresh session)

1. [`IDEA_REPORT.md`](IDEA_REPORT.md) — pipeline output: landscape, ranked ideas, pilot results,
   novelty verdict, critical review, refined-proposal pointer.
2. [`refine-logs/PIPELINE_SUMMARY.md`](refine-logs/PIPELINE_SUMMARY.md) — one-page integration summary.
3. [`refine-logs/FINAL_PROPOSAL.md`](refine-logs/FINAL_PROPOSAL.md) — problem-anchored proposal
   (claims C1–C4, rejected complexity, decision gates).
4. [`refine-logs/EXPERIMENT_PLAN.md`](refine-logs/EXPERIMENT_PLAN.md) — 6 experiment blocks
   (E0–E6), ~14 GPU-days, with **E1 as the hard go/no-go gate**.
5. [`refine-logs/EXPERIMENT_TRACKER.md`](refine-logs/EXPERIMENT_TRACKER.md) — update as runs complete.
6. [`pilot/PILOT_RESULTS.json`](pilot/PILOT_RESULTS.json),
   [`pilot/PILOT_BUDGET_SWEEP.json`](pilot/PILOT_BUDGET_SWEEP.json) — raw pilot numbers.
7. [`LITERATURE_LANDSCAPE.md`](LITERATURE_LANDSCAPE.md) — full literature map.

## Next action — one-click resume

**Pick A for autonomous drive (recommended), B for finer manual control.**

### A. Autonomous drive

Paste this whole prompt as the first message of a fresh session (or invoke it via `/goal`):

```
/goal Drive the GradSpan-KD project to completion per refine-logs/EXPERIMENT_PLAN.md.
Read IDEA_REPORT.md and refine-logs/PIPELINE_SUMMARY.md first to load context — DO NOT
re-run /idea-discovery; it is complete.

Step 1: implement and run E0 (KD gradient spectrum on ResNet-56 → ResNet-20 / CIFAR-100)
via /experiment-bridge. Record the measured intrinsic effective rank R; this calibrates
every later budget.

Step 2: implement and run E1 (parameter-gradient vs late-layer-feature vs input-saliency
space, SAME D-optimal criterion, SAME projected dimension, budgets m ∈ {1R, 2R, 4R}) on
BOTH the vision pair AND a text pair (BERT-base → BERT-small on SST-2). E1 IS A HARD
GO/NO-GO GATE.

Decision after E1:
- If parameter-gradient space beats late-layer-feature space by a clear, multi-seed margin
  on ≥1 modality: proceed to E2 (full rank-calibrated budget sweep with GRAFT and TAGCOS
  as NATIVE baselines, not strawmen), then E3 (KD vs CE relative advantage), E4 (cheap-
  gradient ablation + end-to-end wall-clock including selection cost), E5 (method ablations).
- If grad-space ties feature-space on both modalities: STOP the experiment suite. Do not
  burn E2+ GPU budget. Write a short diagnostic-only note documenting the negative result.

Update refine-logs/EXPERIMENT_TRACKER.md after EVERY run. Use /analyze-results after each
block and /result-to-claim after E2/E4 to judge which of claims C1–C4 the results support.
After experiments are decisive, invoke /paper-writing to draft a submission-ready PDF.

Hard limits: kill any single run exceeding 3× its estimated wall-clock; abort the suite
if total GPU-day spend exceeds 25.

Shared-GPU coordination (mandatory; full pattern in CLAUDE.md > "Shared-GPU coordination"):
- Before EVERY CUDA-using launch: snapshot via `nvidia-smi`, pick a GPU index whose
  `memory.free` is at least 40 GB (or 20 GB for E0 / single-student E1), export
  CUDA_VISIBLE_DEVICES to that index, and log its gpu_uuid in the run record.
- If no GPU meets the threshold, poll every 30s. If polling exceeds 60 minutes, PAUSE the
  experiment suite, note the wait in refine-logs/EXPERIMENT_TRACKER.md, and switch to CPU
  iteration or writing work until a GPU frees.
- Never claim "all GPUs". Multi-GPU runs (E2 parallel-across-seeds) must wait for N free
  GPUs explicitly. Checkpoint every epoch — another tenant's spike can OOM-kill the run.
- Record every long wait (>10 min) and every OOM/kill in the GPU-coordination log in
  EXPERIMENT_TRACKER.md, including GPU UUIDs. Honest end-to-end wall-clock depends on this.

Non-negotiables:
- Drop the falsified claim "KD gradients are more low-rank than CE" — pilot says wash.
- Drop the falsified claim "structural ≠ hard" — pilot correlation is +0.59.
- Report SEED VARIANCE, not just means.
- End-to-end wall-clock MUST include selection cost. The compute-saving claim is decided
  by this number, not by training-only time.
- Position the paper as the FINDING ("right space"), not a new algorithm.
```

### B. Step-by-step (finer control)

```
/experiment-bridge      # implement E0 + E1 code from EXPERIMENT_PLAN.md
/run-experiment         # launch E0 (vision spectrum)
/monitor-experiment     # follow progress
/analyze-results        # confirm R, implement and launch E1 vision + text
# After E1 results:
/result-to-claim        # judge: does C2 hold? gate decision
# If gate passes:
/run-experiment         # E2, E3, E4 in parallel where possible
/auto-review-loop       # iterate the writeup via external review
/paper-writing          # final PDF
```

> ⚠️ **Do NOT run `/research-pipeline`** on a fresh server session — it will redo
> `/idea-discovery` from scratch and discard the existing `IDEA_REPORT.md`. Use the `/goal`
> prompt in section A instead.

## Environment requirements

- Python ≥ 3.10
- **CUDA-enabled PyTorch** (the original pilot environment had only `2.10.0+cpu`; install the
  CUDA wheel for your driver — see `setup_server.sh`).
- `numpy`, `scikit-learn`, `transformers`, `datasets`, `torchvision`, `wandb` (optional).
- **Codex MCP** (optional but recommended for independent cross-model review):
  `claude mcp add codex -s user -- codex mcp-server`

Run `CUDA_TAG=cu121 bash setup_server.sh` after cloning to install everything and verify CUDA.

### Shared-GPU coordination (the server is multi-tenant)

**The target server is shared. A6000 GPUs (≈48 GB each) may be partly or fully occupied by
other users at any time.** Every CUDA-using command MUST check first; never claim "all GPUs",
never assume a specific index is free.

**Before every `python ...` that touches CUDA, do this:**

1. Snapshot current GPU state:
   ```bash
   nvidia-smi --query-gpu=index,name,memory.free,memory.used,utilization.gpu --format=csv
   ```
2. Pick the lowest-index GPU whose `memory.free` is at least the threshold for your job:
   - **E0 spectrum, single-student E1**: need ≥ **20 GB** free.
   - **E2 full distillation, E3, E4, paper-writing fine-tunes**: need ≥ **40 GB** free
     (leaves ~8 GB headroom against another tenant's spike → OOM-kill).
3. Pin to that GPU and record its UUID in your run log:
   ```bash
   export CUDA_VISIBLE_DEVICES=<idx>
   nvidia-smi --query-gpu=gpu_uuid -i <idx> --format=csv,noheader   # log this number
   python experiments/<script>.py
   ```
4. **If no GPU meets the threshold**: poll (`sleep 30; goto 1`). One-liner if you want it:
   ```bash
   while ! nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
       | awk -F, '$2+0 >= 40000 {print $1; found=1; exit} END{exit !found}'; do sleep 30; done
   ```
5. **If polling lasts >60 minutes**: STOP. Update `refine-logs/EXPERIMENT_TRACKER.md` with the
   wait, then switch to CPU iteration on `pilot/`-scale data or writing/analysis work until a
   GPU frees. Do NOT bypass the threshold by running on a near-full GPU — the OOM-kill mid-run
   wastes more time than the wait.

**Other rules:**
- **Pin one GPU per process.** Multi-GPU runs (E2 parallel-across-seeds) must wait for N free
  GPUs explicitly, not assume `torch.cuda.device_count()` is "yours".
- **Checkpoint every epoch.** Another tenant's spike can OOM-kill yours; you should be able to
  resume on a different GPU index without losing >1 epoch of work.
- **Record every long wait (>10 min) and every OOM/kill** in `EXPERIMENT_TRACKER.md`'s
  GPU-coordination log. The paper's end-to-end wall-clock claim depends on honest accounting
  of waits and restarts.

## Key claims (for /result-to-claim, paper writing)

- **C1** — D-optimal coreset over the principal KD-gradient subspace beats random / EL2N /
  GraNd / high-loss / low-loss / CRAIG / LESS-style / DPP at aggressive budgets, and cuts
  seed variance.
- **C2** *(dominant contribution)* — same criterion in parameter-gradient space beats it
  in late-layer-feature and input-saliency space; gap is larger for text than for images.
- **C3** — loss/difficulty-based selection is uniquely harmful for KD; GradSpan-KD's
  *relative* advantage over loss-based selection is larger for KD than for CE.
- **C4** *(optional)* — corrupted teacher labels concentrate in the residual subspace;
  structural-energy score detects them.

## Non-negotiables (carry into every later phase)

- **E1 BEFORE E2.** If E1 ties on both modalities, kill the method claim — don't waste E2 GPU budget.
- **GRAFT and TAGCOS must be NATIVE baselines, not strawmen.**
- **Report seed variance, not just means.**
- **End-to-end wall-clock must INCLUDE selection cost.** The compute-saving claim is decided
  by this number, not by training-only time.
- **Drop "KD more low-rank than CE"** — pilot falsified it. The KD hook is the capacity gap,
  not the spectrum.
- **Drop "structural ≠ hard" as stated** — pilot falsified it (corr +0.59). The replacement
  claim is "loss-based selection fails at aggressive budgets; geometric selection doesn't."
- **Lead with the finding, not the algorithm.** The paper is *"the right space for KD coreset
  selection is parameter-gradient space"*, with GradSpan-KD as the instantiation. Algorithmic
  novelty alone is 5–6/10 (recombination of LESS / GRAFT / TAGCOS / CCS / ICLR-2025-MD).
- **The server is shared; never assume a GPU is yours.** Every CUDA-using `python ...` must be
  preceded by an `nvidia-smi` check, a `CUDA_VISIBLE_DEVICES` pin to a GPU with enough free
  memory (40 GB for full distillation; 20 GB for E0/small-E1), and a logged `gpu_uuid`. If
  polling exceeds 60 minutes, pause the suite and do CPU/writing work. Full pattern is in the
  "Shared-GPU coordination" section above.
