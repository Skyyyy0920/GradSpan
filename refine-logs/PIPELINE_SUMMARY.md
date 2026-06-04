# Pipeline Summary — GradSpan-KD

**Problem**: Knowledge distillation under an aggressive data/soft-label budget — which small
subset of inputs to distill on, and in what geometry to choose it.
**Final Method Thesis**: Keep the subset whose per-sample KD-loss *parameter-gradients* span the
principal gradient subspace (D-optimal coverage); parameter-gradient space is the shared
coordinate system where samples' teacher–student-gap updates are low-rank and comparable.
**Final Verdict**: REVISE — viable as a *finding* paper; one go/no-go experiment (E1) gates it.
**Date**: 2026-05-21

## Final Deliverables
- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`
- Review summary: `IDEA_REPORT.md` §Phase 3 (novelty) and §Phase 4 (critical review)

## Contribution Snapshot
- **Dominant contribution**: the finding — geometric data selection for KD must be done in
  parameter-gradient space; feature-space (GRAFT), input-saliency, and loss-magnitude selection
  measurably underperform.
- **Optional supporting**: rank-calibrated budgeting; ~8× seed-variance reduction; Epiplexity
  structural-energy score.
- **Explicitly rejected complexity**: dynamic per-batch re-selection; curriculum of coresets; a
  load-bearing coreset error-bound theory; any new KD loss.

## Must-Prove Claims
- **C1**: D-optimal coreset over the principal KD-gradient subspace beats random / loss-based /
  CRAIG / LESS-style / DPP at aggressive budgets, and cuts seed variance.
- **C2**: same criterion in parameter-gradient space beats it in feature and saliency space;
  gap larger for text (token mismatch). ← dominant contribution.
- **C3**: loss/difficulty-based selection is uniquely harmful for KD; GradSpan-KD's relative
  advantage over loss-based selection is larger for KD than for CE.
- **C4** (optional): corrupted teacher labels concentrate in the residual subspace.

## First Runs to Launch
1. E0 — KD-gradient spectrum on ResNet-56→ResNet-20 / CIFAR-100; measure intrinsic rank R.
2. E1 — vision space head-to-head (gradient vs feature vs saliency) at m ∈ {1R, 2R, 4R}.
3. E1 — text space head-to-head (BERT pair); the token-mismatch test.

## Main Risks
- **GRAFT feature-space ties gradient-space** → E1 is the go/no-go gate; pivot to a diagnostic
  note or kill if it ties on both modalities.
- **Selection cost not amortized** → E4 cheap-gradient ablation + end-to-end wall-clock; motivate
  via saved teacher soft-label inference.
- **KD not provably special** → if C3 fails, reframe as a general aggressive-budget coreset paper.
- **Pilot was CPU / toy-scale only** → every pilot conclusion must be re-tested at E0–E2 scale.

## Pipeline-Level Caveat
The Codex MCP (`gpt-5.4`) external-reviewer backend was unavailable; brainstorming, novelty
cross-check, and critical review were done in-session by Claude Opus 4.7. Install the Codex CLI
and re-run `/novelty-check` and `/research-review` for an independent cross-model check before
committing to the full ~14 GPU-day suite.

## Next Action
- Proceed to `/run-experiment` — launch E0, then the E1 gate.
