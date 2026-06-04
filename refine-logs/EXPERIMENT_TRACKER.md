# Experiment Tracker — GradSpan-KD

**Date created**: 2026-05-21
Update the Status column as runs complete. Blocks gated by E1 must not start until E1 passes.

| Block | Run | Setup | Supports | Status | Result / notes |
|-------|-----|-------|----------|--------|----------------|
| E0 | gradient spectrum | ResNet-56→ResNet-20 / CIFAR-100 | C1, C2 | ☐ not started | measure intrinsic rank R |
| E0 | gradient spectrum | BERT-base→BERT-small / SST-2 | C1, C2 | ☐ not started | measure R for text |
| E1 ★GATE | space head-to-head | vision, m∈{1R,2R,4R} | C2 | ☐ not started | grad vs feature vs saliency |
| E1 ★GATE | space head-to-head | text, m∈{1R,2R,4R} | C2 | ☐ not started | token-mismatch test |
| E2 | budget sweep + baselines | vision | C1 | ⛔ blocked by E1 | incl. GRAFT, TAGCOS native |
| E2 | budget sweep + baselines | text | C1 | ⛔ blocked by E1 | incl. GRAFT, TAGCOS native |
| E3 | KD vs CE relative advantage | both | C3 | ⛔ blocked by E1 | restores KD-special claim |
| E4 | cheap-gradient ablation | vision | efficiency | ⛔ blocked by E1 | full/last-layer/one-step/proxy |
| E4 | end-to-end wall-clock | both | efficiency | ⛔ blocked by E1 | incl. selection + teacher inference |
| E5 | projection ablation | vision | method | ⛔ blocked by E1 | Count Sketch vs JL vs none |
| E5 | selection-rule ablation | vision | method | ⛔ blocked by E1 | D-optimal vs leverage vs quota |
| E6 | label-noise injection | vision | C4 (optional) | ⛔ blocked by E1 | Epiplexity falsification |

## Decision Log
- 2026-05-21 — Pilot (CPU, sklearn-digits): low-rank CONFIRMED (eff. rank 23/1200); D-optimal coreset
  +8.8 pts over random at budget≈rank with 8× lower variance; "KD more low-rank than CE" and
  "structural ≠ hard" FALSIFIED. Project repositioned as a finding paper. See `IDEA_REPORT.md`.

## Status legend
☐ not started · ⏳ running · ✅ done · ⛔ blocked · ❌ failed/killed
🕐 waiting for GPU (server full) — record start time + threshold used

## GPU-coordination log
Record every long GPU wait (>10 min) and every OOM/kill caused by tenant displacement, with
GPU UUID and approximate competing-tenant memory footprint. This is for honest end-to-end
wall-clock accounting in the paper's efficiency section.

| Date / time | Block / run | GPU index | GPU UUID | Event | Notes |
|-------------|-------------|-----------|----------|-------|-------|
| _example_   | E2 seed 3   | 2         | GPU-xxxx | OOM-kill at epoch 47 | another job spiked to 44GB; restarted on idx 5 |
