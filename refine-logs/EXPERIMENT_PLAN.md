# Experiment Plan — GradSpan-KD

**Date**: 2026-05-21
**Anchored to**: `refine-logs/FINAL_PROPOSAL.md` (claims C1–C4)
**Total budget estimate**: ~14 GPU-days on a 3090/4090-class cluster
**Philosophy**: do not run the full suite on an unproven thesis. Block E1 is a hard go/no-go gate.

## Setups (fixed across all blocks)

- **Vision KD pair**: ResNet-56 → ResNet-20 on CIFAR-100; also WRN-40-2 → WRN-16-2.
- **Text KD pair**: BERT-base → BERT-small (or DistilBERT-size) on a GLUE task (SST-2/MNLI);
  GPT-2-medium → GPT-2-small on a small LM corpus for the token-mismatch test.
- **Budget axis**: rank-calibrated — first measure intrinsic effective rank `R` of the KD-gradient
  matrix (E0), then sweep coreset size m ∈ {0.5R, 1R, 2R, 4R, 8R, 0.5N, N}.
- **Seeds**: 5 per cell (variance is a reported metric, not just a confidence interval).
- **Metric**: student test accuracy/score, its std across seeds, and end-to-end wall-clock.

## Shared-GPU coordination (the server is multi-tenant)

A6000s (≈48 GB each) on this server are shared. **Before every CUDA-using `python ...`:**

1. Snapshot:
   ```bash
   nvidia-smi --query-gpu=index,name,memory.free,memory.used,utilization.gpu --format=csv
   ```
2. Pick the lowest-index GPU whose `memory.free` is above the block's threshold (table below),
   then pin and log it:
   ```bash
   export CUDA_VISIBLE_DEVICES=<idx>
   nvidia-smi --query-gpu=gpu_uuid -i <idx> --format=csv,noheader   # log this in the run record
   ```
3. If no GPU meets the threshold, poll every 30 s. One-liner if useful:
   ```bash
   while ! nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
       | awk -F, '$2+0 >= 40000 {print $1; found=1; exit} END{exit !found}'; do sleep 30; done
   ```
4. If polling exceeds 60 min, **pause the suite**, write the wait into
   `EXPERIMENT_TRACKER.md`'s GPU-coordination log, and do CPU iteration / writing instead.

**Per-block GPU plan:**

| Block | Concurrency | Min free / GPU | Strategy if box is full |
|-------|-------------|----------------|--------------------------|
| E0    | 1 GPU       | 20 GB          | Wait — block is short (~30 min) |
| E1    | 1 GPU per modality (can serialize) | 20 GB | Serialize vision then text |
| E2    | up to N GPUs (parallel seeds) | 40 GB | Reduce parallelism, serialize seeds |
| E3    | 1–2 GPUs    | 40 GB          | Serialize KD vs CE legs |
| E4    | 1 GPU       | 40 GB          | Wait |
| E5    | up to N GPUs | 20–40 GB      | Reduce parallelism |
| E6    | 1 GPU       | 20 GB          | Wait |

Do NOT bypass the threshold by running on a near-full GPU — an OOM-kill mid-run costs more
than the wait. Checkpoint every epoch so a kill loses ≤ 1 epoch of work.

## Experiment Blocks

### E0 — Diagnostic backbone  (~0.5 GPU-day)
*Supports C1, C2; calibrates the budget axis.*
- Compute per-sample KD-gradient matrices for each pair; Count Sketch project; SVD.
- Report effective rank, stable rank, spectral decay; KD vs CE spectrum (honest — pilot says wash).
- Output: the rank `R` used to calibrate every later budget.

### E1 — Space head-to-head  ★ GO/NO-GO GATE ★  (~2 GPU-days)
*Supports C2 (dominant contribution).*
- Fix the D-optimal criterion and projected dimension. Build coresets in three spaces:
  parameter-gradient / late-layer-feature / input-saliency. Retrain student; compare at m ∈ {1R,2R,4R}.
- Run on **both** vision and text (text is where the token-mismatch mechanism is real).
- **Decision gate**: parameter-gradient space must beat feature space by a clear, multi-seed margin
  on ≥1 modality (target: both). **If it ties on both → STOP.** Pivot to a diagnostic-only note or
  kill the project. Do not run E2–E6 until this passes.

### E2 — Main anchor result  (~5 GPU-days)
*Supports C1.*
- Full rank-calibrated budget sweep. Baselines (all native, no strawmen): random, EL2N, GraNd,
  high-loss, low-loss, CRAIG (gradient matching), **GRAFT** (feature MaxVol), **TAGCOS** (gradient
  clustering), LESS-style similarity, DPP-on-embeddings.
- Report accuracy AND seed variance. Both modalities, both pairs.
- **Decision gate**: GradSpan-KD must beat GRAFT and TAGCOS at aggressive budgets. If not, the
  method has no algorithmic novelty — downgrade to the diagnostic note from E1.

### E3 — KD relevance  (~2 GPU-days)
*Supports C3.*
- Same selection methods, applied to (a) KD/KL training and (b) plain CE training of the same
  student. Compare the *relative* advantage of GradSpan-KD over loss-based selection in each.
- **Decision gate**: if the relative advantage is indistinguishable between KD and CE, drop the
  "KD-special" claim and reframe as a general aggressive-budget coreset paper (weaker).

### E4 — Efficiency / amortization  (~1.5 GPU-days)
*Defends the practical claim.*
- Cheap-gradient ablation: full / last-layer / LoRA-only / one-step / logit-difference proxy.
- End-to-end wall-clock INCLUDING selection cost and teacher soft-label inference.
- Report the break-even point: number of coreset epochs / number of distilled students at which
  selection cost is amortized.

### E5 — Method ablations  (~2 GPU-days)
- Projection: Count Sketch vs Rademacher JL vs none. Selection rule: greedy D-optimal vs
  ridge-leverage-score vs per-PC quota. Subspace dimension k. Robustness to teacher choice / temp.

### E6 — Epiplexity falsification (optional)  (~1 GPU-day)
*Supports C4.*
- Corrupt teacher logits for a known fraction of samples; verify they concentrate in the residual
  subspace (low structural-energy score). Decisive cheap test of the structural/idiosyncratic split.

## Run Order

```
E0  ──▶  E1 (GATE)  ──▶  { E2 ∥ E3 ∥ E4 }  ──▶  E5  ──▶  E6 (optional)
                 │
                 └─ fail ─▶ pivot to diagnostic note OR kill
```

## Results-to-Claims Matrix

| Outcome of E1 | Outcome of E2 | Allowed claim |
|---------------|---------------|---------------|
| grad ≫ feature, both modalities | GradSpan-KD beats GRAFT/TAGCOS | Full paper: "the right space is parameter-gradient space + GradSpan-KD method" |
| grad ≫ feature on 1 modality only | beats baselines | Scoped paper: finding holds for that modality; honest limitation |
| grad ≈ feature everywhere | — | Kill the method claim; at most a short negative/diagnostic note |
| E1 passes | GradSpan-KD ≈ GRAFT | Diagnostic paper: "space matters" finding only, no method claim |
| any | C3 fails | Drop "KD-special"; reframe as general coreset (second-tier venue) |

## First 3 Runs to Launch
1. **E0** on ResNet-56→ResNet-20 / CIFAR-100 — measure `R`, calibrate budgets.
2. **E1** vision space head-to-head at m ∈ {1R, 2R, 4R}.
3. **E1** text space head-to-head (BERT pair) — the token-mismatch test.

Do not launch E2 until both E1 runs clear the gate.
