# Literature Landscape — Gradient-Geometry Data Selection for Knowledge Distillation

**Date**: 2026-05-21
**Direction**: Select training data for KL knowledge distillation by the geometry of per-sample
student-parameter gradients — Count Sketch (JL) projection → PCA/SVD → D-optimal coreset that
spans the principal gradient subspace.

## 1. Landscape Map (sub-directions)

### A. Gradient-based data selection / attribution (the core competitive zone)
| Paper | Venue/Year | Method | Relation to us |
|-------|-----------|--------|----------------|
| **LESS** (Xia et al.) | ICML 2024 | Per-sample **LoRA gradients** + **random (JL) projection** to low-dim datastore; select by **cosine similarity to a target few-shot gradient** | **Closest pipeline.** Same gradient→project→select skeleton. Differs: target-task similarity, NOT PCA-subspace coreset; instruction tuning, NOT KD. |
| **GRAFT** (Jha et al.) | arXiv 2508.13653, Aug 2025 | Per-batch low-rank feature representation + **Fast MaxVol** sampler picks subset spanning the **dominant subspace**; adaptive size via gradient-approx | **Closest selection criterion.** MaxVol ≈ D-optimal / max-volume. Differs: in-training per-batch, feature decomposition, not a global one-shot gradient-space PCA coreset; not KD. |
| TRAK (Park et al.) | ICML 2023 | Random projection (FJLT/JL) of gradients → datamodel attribution | Establishes JL gradient projection; attribution, not coreset. |
| GraSS / Gradient Sketches | arXiv 2505.18976, 2402.03994 | Count Sketch / sparse projection of gradients for scalable attribution | Confirms Count Sketch on gradients is established — for *attribution*, not KD coreset. |
| DsDm | arXiv 2401.12926 | Datamodels-based dataset selection to maximize target performance | Model-aware selection; regression-based, not gradient-geometry. |
| Influence Distillation / In2Core | arXiv 2505.19051 / 2408.03560 | Influence functions / internal gradients → coreset for instruction tuning | Influence-based coreset; ~50% data, 3.5x faster selection. Not subspace geometry. |
| CRAIG / GRAD-MATCH | ICML 2020 / 2021 | Coreset whose **summed gradient matches the full-batch gradient** | Different criterion: gradient *matching*, not subspace *spanning*/D-optimal. |
| GradPCA | arXiv 2505.16017 | PCA on gradients (NTK alignment) for **OOD detection** | PCA-on-gradients exists — but for OOD, not data selection. |
| Uncertainty-Aware Gradient SNR selection | arXiv 2601.13697 | Gradient signal-to-noise ratio for instruction-tuning selection | Signal/noise framing close in spirit to structural/idiosyncratic split. |

### B. Low-rank structure of fine-tuning gradients (the core assumption)
- **GaLore** (arXiv 2403.03507): periodic SVD on gradients reveals a slow-changing low-rank subspace; full-parameter training inside it. Empirically validates "fine-tuning gradients are low-rank in parameter space."
- **LoRA / intrinsic dimension**: weight *updates* are low-rank; per-example LoRA gradients ≈ samples from a posterior over low-rank updates.
- Q-GaLore, WeLore, FFT-based dynamic subspace selection: all rely on the empirical low-rank gradient subspace.

### C. KD-specific data / sample treatment
- "Improve KD via Label Revision and Data Selection" (arXiv 2404.03693): selects samples where teacher supervision is *reliable* — quality-driven, not geometry-driven.
- AdaKD, curriculum KD, self-paced KD: re-weight samples by **difficulty**, not select a coreset.
- **Key KD finding (capacity gap)**: hard-sample mining *hurts* KD — the student cannot absorb teacher knowledge on hard samples. Sample weighting in KD is biased toward *easy* samples.
- KD spectral-bias theory (2025): KD induces an *exaggerated spectral bias* — the student's parameter trajectory converges faster along top data eigendirections. **This directly supports the low-rank shared-gradient hypothesis for the KD loss specifically.**
- KD as partial variance reduction (arXiv 2305.17581): soft targets reduce gradient variance across examples — another reason KD gradients should be *more* correlated/low-rank than CE gradients.

### D. Diversity / volume-based selection
- DPP-based selection (P3, DQO, Reliability-Aware DPP): maximize **log-det of a similarity kernel** = volume = diversity. Mathematically the same family as D-optimal/log-det. Applied to embeddings/responses, **not gradient subspaces**.

### E. Information-theoretic framing
- **Epiplexity** (arXiv 2601.03220, "From Entropy to Epiplexity"): formal split of data into **structural information** (learnable by a compute-bounded model) vs **time-bounded entropy** (idiosyncratic randomness). The paper explicitly positions epiplexity as a guide for data selection. Natural theoretical home for "principal subspace = structural / residual = idiosyncratic."

## 2. Structural Gaps Identified

1. **KD × gradient-geometry coreset is empty.** Gradient-based selection (LESS, GRAFT, TRAK) targets CE/instruction-tuning losses or attribution. KD data selection (label revision, curriculum) is quality/difficulty-driven. Nobody selects a KD coreset from the geometry of the **KL-loss gradient**.
2. **PCA-subspace + D-optimal coreset is not the standard criterion.** LESS uses target similarity; GRAFT uses per-batch MaxVol on features; CRAIG/GRAD-MATCH use gradient matching. A *global, one-shot* PCA of the projected KD-gradient matrix → D-optimal coreset spanning the principal subspace is a distinct recipe.
3. **The structural-vs-idiosyncratic decomposition is unformalized.** "Principal subspace = transferable structure, residual = sample-specific noise" is intuitive but never tied to a measurable selection rule, nor to Epiplexity.
4. **The space matters but nobody isolates it.** The claim that gradient/parameter space is low-rank but saliency/input space is *not* (because samples have different tokens) is a clean, testable diagnostic — and would explain why feature-space methods (GRAFT-style) may underperform here.
5. **The KD capacity-gap tension is unexplored.** "Hard samples hurt KD" vs "select informative samples" — structural samples are not the same as hard samples. Whether structural selection avoids the capacity-gap failure mode is an open and interesting question.

## 3. Closest Threats (for deep novelty check)
- **LESS** — shares gradient→JL-project→select skeleton. Differentiate on: PCA-subspace D-optimal criterion (vs target similarity), KD/KL loss, task-agnostic.
- **GRAFT** — shares "select subset spanning a dominant subspace via max-volume." Differentiate on: global gradient-space PCA (vs per-batch feature subspace), one-shot coreset, KD, and the gradient-vs-saliency-space ablation as a *finding*.
- **DPP selection** — shares log-det objective. Differentiate on: log-det over *gradient principal coordinates* (vs embedding similarity kernel).

## 4. Implications for Idea Generation
- The defensible novelty is the **conjunction**: KD/KL loss + global gradient-space PCA + D-optimal coreset + the structural/idiosyncratic (Epiplexity) interpretation + the gradient-vs-saliency-space negative result.
- The strongest standalone contribution may be **diagnostic**: "KD gradients are empirically low-rank across samples; saliency errors are not" — a finding that is publishable regardless of how well the coreset performs.
- Must address the KD capacity-gap finding head-on (structural ≠ hard).
