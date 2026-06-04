# GradSpan-KD

Data selection for knowledge distillation by the geometry of the student's per-sample KD-loss gradient.

> **TL;DR.** For each training example, take the gradient of the KL distillation loss w.r.t. the student parameters. Compress with Count Sketch, PCA the per-sample gradient matrix, and keep a D-optimal coreset that spans the principal subspace. Positioned as a *finding paper*: **for KD, geometric data selection must be done in parameter-gradient space — feature-space (GRAFT-style), saliency-space, and loss-magnitude selection measurably underperform.**

## Status

Phases 1–4.5 of the research pipeline are complete (literature → ideas → CPU pilot → novelty check → critical review → refined proposal + experiment plan). A small CPU pilot confirms the core mechanism:

- Per-sample KD-loss gradients are strongly low-rank (effective rank ≈ 23 of 1200 ambient dims).
- A D-optimal subspace-spanning coreset beats random selection by **+8.8 accuracy points with ~8× lower seed variance** at aggressive compression (coreset size ≈ 1–2× the intrinsic gradient rank); the advantage vanishes at modest pruning rates.
- Two sub-hypotheses were falsified by the pilot ("KD more low-rank than CE"; "structural ≠ hard") and have been dropped/reframed — see `IDEA_REPORT.md`.

**Next step:** run **E1** from `refine-logs/EXPERIMENT_PLAN.md` — a gradient-space vs. feature-space (GRAFT) head-to-head that gates the project.

## Repository contents

| Path | What it is |
|------|------------|
| [`CLAUDE.md`](CLAUDE.md) | Project context, auto-loaded by Claude Code at session start. Contains the one-click resume command. |
| [`IDEA_REPORT.md`](IDEA_REPORT.md) | Full pipeline output: landscape, ranked ideas, pilot results, novelty verdict, critical review. |
| [`LITERATURE_LANDSCAPE.md`](LITERATURE_LANDSCAPE.md) | Detailed literature map with closest competitors. |
| [`refine-logs/`](refine-logs/) | Refined proposal, experiment plan, run tracker, pipeline summary. |
| [`pilot/`](pilot/) | CPU diagnostic pilots + JSON results. |
| `setup_server.sh` | Install CUDA PyTorch + dependencies; verify GPU; re-run pilot as a smoke test. |
| `requirements.txt` | Non-CUDA Python dependencies. |

## Setup on a GPU server

```bash
git clone https://github.com/Skyyyy0920/GradSpan.git
cd GradSpan
CUDA_TAG=cu121 bash setup_server.sh    # set CUDA_TAG to match your driver
claude                                  # opens Claude Code in this directory
```

`CLAUDE.md` is auto-loaded; follow the "Next action — one-click resume" section to drive the project to completion.

## Method (one paragraph)

For each example, compute `g_i = ∇_θ KL(teacher ‖ student)` — the KD-loss gradient w.r.t. the student parameters. Project each `g_i` into a low-dimensional space with Count Sketch (a Johnson–Lindenstrauss distance-preserving projection). Stack the projected gradients into an N×d matrix, SVD it, and take the top-k right singular vectors as the principal gradient subspace. Greedily build a coreset by maximizing `log det(I + G_S^⊤ G_S / λ)` over the principal coordinates (D-optimal design). Distill the student only on this coreset. The supporting hypothesis is that fine-tuning gradients are low-rank in parameter space because all examples share one parameter basis and the same teacher–student gap; the empirically novel claim is that they are *more* low-rank in parameter space than in feature or saliency space, which is why doing the same selection in those spaces (à la GRAFT) underperforms.

## Citation

(Placeholder — paper not yet submitted.)

```bibtex
@unpublished{gradspan_kd_2026,
  title  = {GradSpan-KD: Parameter-Gradient-Space Coreset Selection for Knowledge Distillation},
  author = {Huang, Tianhao},
  year   = {2026},
  note   = {Working draft}
}
```
