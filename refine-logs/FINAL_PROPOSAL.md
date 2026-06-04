# Final Proposal — GradSpan-KD

**Date**: 2026-05-21
**Status**: REVISE (viable; one make-or-break experiment gates the project)

## Problem Anchor (frozen — do not drift)

> **Knowledge distillation under an aggressive data/soft-label budget: when you can afford to
> distill the student on only a small subset of the available inputs, which subset should it be,
> and in what geometry is that subset best chosen?**

This anchor is fixed. Scope creep to "general coreset selection", "dynamic in-training selection",
or "a new KD loss" is out of bounds for the main paper.

## Final Method Thesis (one sentence)

For knowledge distillation under aggressive data budgets, keep the subset whose per-sample
**KD-loss gradients with respect to the student parameters** span the principal gradient subspace
(D-optimal / log-det coverage) — because **parameter-gradient space, unlike feature, saliency, or
input space, is the shared coordinate system in which different samples' "teacher–student gap"
updates are low-rank and directly comparable.**

## Dominant Contribution

A **finding**: *geometric data selection for KD must be performed in parameter-gradient space.*
Selecting in feature space (à la GRAFT), input-saliency space, or by loss magnitude
(à la the ICLR-2025 medium-difficulty KD method) measurably underperforms. **GradSpan-KD** — Count
Sketch projection → PCA of the per-sample KD-gradient matrix → greedy D-optimal coreset over the
principal subspace — is the *instantiation* of that finding, not the headline.

## Supporting Contributions (kept small)

- **Rank-calibrated budgeting**: the coreset budget at which geometric selection helps is
  predictable from the *measured* intrinsic effective rank of the KD-gradient matrix.
- **Reproducibility**: at aggressive budgets random selection is a lottery; D-optimal selection
  cuts seed variance ~8× (pilot).
- **Epiplexity instantiation** (optional): a per-sample structural-information score = gradient
  energy inside the principal subspace; corrupted teacher labels should fall in the residual
  subspace.

## Complexity Intentionally Rejected

| Rejected | Why |
|----------|-----|
| Dynamic per-batch re-selection (GRAFT-style) | Keeps one dominant contribution; one-shot global selection is the claim. Per-batch is GRAFT's territory. |
| Curriculum of coresets (re-select every T epochs) | Interesting extension; excluded from the main paper to avoid diluting the thesis. |
| A coreset error-bound theory as a core contribution | Demoted to optional appendix — CCS already owns the qualitative high-pruning phenomenon; an NTK-regime bound is a nice-to-have, not load-bearing. |
| A new KD loss / temperature scheme | Out of anchor — this is a *what to train on* paper, not a *how to train* paper. |

## Why It Is Not Subsumed by Prior Work

- **GRAFT** (2508.13653): feature space, per-batch, no KD. GradSpan-KD claims — and must show —
  that GRAFT's feature space is the *wrong* space for KD selection.
- **LESS** (ICML 2024): target-similarity influence ranking, no subspace step, CE instruction
  tuning. Different criterion, different task.
- **TAGCOS** (2407.15235): clusters gradients (not principal-subspace D-optimal), instruction
  tuning. GradSpan-KD spans the subspace rather than covering clusters, and targets KD.
- **CCS** (ICLR 2023): coverage in importance-score space. GradSpan-KD's coverage is of the
  gradient principal subspace, rank-calibrated, KD-specific.
- **ICLR-2025 Medium-Difficulty KD**: gradient *magnitude* + difficulty. GradSpan-KD uses gradient
  *direction/geometry* and is not loss-based — which is the point (loss-based selection is uniquely
  harmful in KD).

## Key Claims (carried verbatim into the experiment plan)

- **C1 — Main anchor result.** A D-optimal coreset over the principal KD-gradient subspace beats
  random, loss/difficulty-based, EL2N/GraNd, CRAIG, LESS-style, and DPP-on-embeddings selection at
  aggressive budgets (≤ ~5%; budget ≈ 1–4× the intrinsic gradient rank), and reduces seed variance.
- **C2 — Space asymmetry (dominant contribution).** The *same* D-optimal criterion applied in
  parameter-gradient space beats it applied in late-layer-feature space and input-saliency space;
  the gap is larger for text than for images (token mismatch breaks the shared input basis).
- **C3 — KD relevance.** Loss/difficulty-based selection is uniquely harmful for KD; GradSpan-KD's
  *relative* advantage over loss-based selection is larger for KD than for plain CE training.
- **C4 — Epiplexity (optional).** Injected teacher-label corruption concentrates in the residual
  (idiosyncratic) subspace; the structural-energy score detects it.

## Must-Run Ablations

- Gradient source: full / last-layer / LoRA-only / one-step / logit-difference proxy (gates the
  efficiency claim — selection must be cheaper than the training it saves).
- Projection: Count Sketch vs Rademacher JL vs no projection.
- Selection rule: greedy D-optimal vs ridge-leverage-score sampling vs per-PC quota.
- Subspace dimension k and budget sweep (rank-calibrated).
- End-to-end wall-clock *including* selection and teacher soft-label inference.

## Remaining Risks

| Risk | Mitigation / decision gate |
|------|----------------------------|
| GRAFT feature-space ties gradient-space at scale | **Decision gate — run first.** If C2 fails on both modalities, kill or pivot to a pure diagnostic note. |
| Selection cost not amortized | Cheap-gradient ablation; frame benefit as saved teacher soft-label inference. |
| Text asymmetry weaker than predicted | Still publishable as an image finding; report honestly. |
| KD-vs-CE advantage indistinguishable | Fall back to "general aggressive-budget coreset" framing (weaker, second-tier venue). |

## Verdict

**REVISE → conditionally proceed.** The repositioned finding-paper is viable. Run the C2 space
head-to-head **first** as a go/no-go gate before committing to the full experiment suite.
