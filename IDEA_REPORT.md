# Idea Discovery Report

**Direction**: Knowledge-distillation improvements almost all act on *how* to train (loss form,
sampling weights, noise injection) and treat the dataset as uniform. This direction does data
selection from **gradient geometry**: per-sample gradient of the KL distillation loss w.r.t. the
*student* parameters → Count Sketch (JL) projection → PCA/SVD of the projected per-sample gradient
matrix → select a D-optimal coreset that spans the principal gradient subspace, then continue
training on the coreset. Core assumption: fine-tuning gradients are empirically low-rank in
parameter space (LoRA-style). Key claimed negative finding: the same PCA in saliency (input)
space fails. Connects to the Epiplexity framework (structural vs random information).

**Date**: 2026-05-21
**Pipeline**: research-lit → idea-creator → novelty-check → research-review → research-refine-pipeline
**Generated**: 12 ideas brainstormed → 6 survived filtering → 3 piloted (CPU diagnostic) → 1 recommended

> **Tooling note**: The Codex MCP (`gpt-5.4`) external-reviewer backend was **unavailable in this
> environment** (`spawn codex ENOENT` — the Codex CLI is not installed). Brainstorming, devil's
> advocate, and critical review were therefore performed by the in-session model (Claude Opus 4.7)
> rather than an OpenAI model. The pipeline still produced every deliverable. **Recommended:**
> install the Codex CLI and re-run `/novelty-check` and `/research-review` for an independent
> cross-model check before committing to a full project.

---

## Executive Summary

The core mechanism is **real and the cheap CPU pilot confirms it**: per-sample KD-gradient matrices
are strongly low-rank (effective rank ≈ 23 of 1200 ambient dims), and a D-optimal coreset that
spans the principal gradient subspace **beats random selection by +8.8 accuracy points with an 8×
variance reduction — but only at aggressive compression** (coreset size ≈ 1–2× the intrinsic
gradient rank). At the modest "save ~1/3 compute" budgets in the original framing, the advantage
vanishes. **Recommendation: pursue the idea, but re-anchor it on aggressive compression +
reproducibility, not modest pruning.** Two sub-hypotheses were falsified by the pilot and should be
dropped or reframed (see Idea 1b and Idea 7).

Phase-3 novelty verification then found the method is a **recombination of existing components**
(JL gradient projection — LESS/TRAK; D-optimal/volume selection — GRAFT; task-agnostic gradient
coreset — TAGCOS; high-pruning coverage — CCS; KD gradient-based selection — ICLR 2025). Overall
novelty is **5–6/10**. The honest, defensible paper is therefore **not** "a new coreset algorithm"
but a **finding paper**: *for KD, geometric data selection must be done in parameter-gradient
space — feature/saliency/loss-magnitude spaces measurably underperform* — with **GradSpan-KD** as
the instantiation. The single make-or-break experiment (E1, the space head-to-head vs GRAFT) gates
the whole project; see `refine-logs/EXPERIMENT_PLAN.md`. Final verdict: **REVISE → conditionally
proceed.**

---

## Literature Landscape

See `LITERATURE_LANDSCAPE.md` for the full map. Summary:

- **Closest prior work.** *LESS* (ICML 2024) shares the exact `gradient → JL-project → select`
  skeleton but selects by similarity to a target-task gradient (not a PCA-subspace coreset), for
  instruction tuning, not KD. *GRAFT* (arXiv 2508.13653, Aug 2025) shares the "select a subset
  spanning a dominant subspace via max-volume (≈ D-optimal)" idea but operates per-batch, in-
  training, on *feature* decomposition, not a global one-shot gradient-space PCA, and not KD.
- **The assumption is well-supported.** GaLore and the LoRA literature establish that fine-tuning
  gradients are empirically low-rank in parameter space. 2025 KD theory (exaggerated spectral bias;
  KD as partial variance reduction) predicts KD/KL gradients should be at least as low-rank as CE.
- **KD data selection exists but is quality/difficulty-driven** (label revision, curriculum KD),
  never gradient-geometry-driven. A robust KD finding — *hard-sample mining hurts KD* — must be
  engaged head-on.
- **Epiplexity** (arXiv 2601.03220) is a real, recent framework splitting data into structural vs
  time-bounded-entropy information; it explicitly motivates data selection.
- **Gaps:** (1) KD × gradient-geometry coreset is empty; (2) global one-shot PCA→D-optimal is a
  distinct criterion vs LESS / GRAFT / CRAIG / DPP; (3) the structural-vs-idiosyncratic split is
  unformalized; (4) nobody isolates *why* gradient space is low-rank but saliency space less so;
  (5) the KD capacity-gap tension is unexplored.

---

## Pilot Experiment Results (CPU diagnostic, sklearn-digits KD)

Environment has **no CUDA-enabled PyTorch** (CPU-only build) — full LM/vision-scale pilots were not
possible. A small teacher→student MLP distillation pilot (`pilot/pilot_gradgeom.py`,
`pilot/pilot_budget_sweep.py`) was run instead to get cheap signal on the *mechanism*. Total
compute: ~90 s CPU (0 GPU-hours of the 8 h budget). **These results must be re-tested at LM scale.**

| Test | Result | Verdict |
|------|--------|---------|
| KD per-sample gradient matrix low-rank? | eff. rank **23.4** / 1200; top-5 PCs = 54% energy | **CONFIRMED (strong)** |
| KD gradients *more* low-rank than CE? | norm-eff-rank 0.0195 vs 0.0206 | **NOT confirmed (wash)** |
| Param-gradient space lower norm-rank than input-saliency? | 0.0195 vs 0.149 (7.6×) | **Supported** |
| Structural energy uncorrelated with sample loss? | corr = **+0.59** (high-loss = more structural) | **Falsified** |
| D-optimal coreset > random at 50% budget? | 0.9707 vs 0.9724 | **No (wash/slight loss)** |
| D-optimal coreset > random at budget ≈ rank? | **0.909 ±0.01 vs 0.822 ±0.08 (+8.8 pts, 8× lower variance)** | **POSITIVE** |
| Saliency-space coreset vs gradient-space coreset | 0.967 vs 0.971 | gradient space wins |
| Hard-sample (high-loss) coreset | 0.952 (worst) | "hard hurts KD" reproduced |

**Phase transition (the key finding):** D-optimal's advantage is large and *reliable* when budget
≈ 1–2× the intrinsic gradient rank, shrinks to zero by ~4×, and slightly reverses beyond ~8×
(greedy log-det over-weights boundary samples once coverage is trivial). This sharpens the whole
project: the method is for **aggressive** compression, where it also delivers a second, independent
benefit — **variance reduction** (random selection is a lottery at tiny budgets; D-optimal is not).

Budget sweep (mean test acc ± std over 5 seeds; gradient rank ≈ 23.4):

| Budget | D-optimal | Random | easy-lowloss | Lift (D-opt − rand) |
|--------|-----------|--------|--------------|---------------------|
| 24  | 0.909 ±0.010 | 0.822 ±0.081 | 0.551 ±0.058 | **+0.088** |
| 48  | 0.922 ±0.008 | 0.905 ±0.017 | 0.804 ±0.024 | +0.017 |
| 96  | 0.947 ±0.004 | 0.941 ±0.007 | 0.921 ±0.015 | +0.006 |
| 192 | 0.953 ±0.006 | 0.963 ±0.004 | 0.937 ±0.007 | −0.010 |
| 384 | 0.966 ±0.003 | 0.972 ±0.004 | 0.968 ±0.003 | −0.006 |
| 600 | 0.974 ±0.003 | 0.972 ±0.002 | 0.974 ±0.004 | +0.002 |

---

## Recommended Ideas (ranked)

### 🏆 Idea 1: GradSpan-KD — Aggressive Coreset Distillation by Spanning the Principal Gradient Subspace — RECOMMENDED
- **Hypothesis**: At aggressive compression (coreset size near the intrinsic effective rank of the
  per-sample KD-gradient matrix), a D-optimal coreset that spans the principal gradient subspace
  substantially and *reliably* outperforms random, loss-based, and saliency-space selection; the
  advantage decays predictably to zero as budget exceeds ~4× the rank.
- **Method**: per-sample ∇_θ KL(teacher‖student) → Count Sketch projection → PCA → greedy D-optimal
  (log-det) selection over the top-k principal coordinates → distill on the coreset. Includes the
  diagnostic backbone (gradient-matrix spectrum; gradient-space vs saliency-space comparison).
- **Minimum experiment**: 2–3 KD pairs (e.g., ResNet-56→ResNet-20 on CIFAR-100; BERT/GPT-2
  teacher→small student on a text dataset). Sweep coreset budget 1–25× the measured gradient rank.
  Baselines: random, EL2N/GraNd, high/low-loss, CRAIG (gradient matching), LESS-style similarity,
  GRAFT-style feature-MaxVol, DPP-on-embeddings. Metrics: student accuracy *and* its variance at
  fixed budget.
- **Expected outcome**: positive — clear win at aggressive budgets + variance reduction; honest
  null at large budgets (this scoping is itself a contribution).
- **Novelty**: see Phase 3. Closest: LESS (different criterion, not KD), GRAFT (per-batch feature
  MaxVol, not global gradient PCA, not KD).
- **Feasibility**: vision pilot ≤ 1 GPU-day; text pilot ≤ 2 GPU-days. Selection cost (per-sample
  gradients) is the main risk to net savings — see Idea 6.
- **Risk**: MEDIUM. The mechanism is pilot-confirmed; risk is whether the aggressive-budget win
  survives at LM scale and whether selection cost is amortized.
- **Contribution type**: method + diagnostic.
- **Pilot result**: POSITIVE (+8.8 pts, 8× variance reduction at budget ≈ rank).
- **Reviewer's likely objection**: "GRAFT/LESS already do gradient-subspace selection." Rebuttal:
  global one-shot gradient-space PCA + D-optimal for the *KD/KL loss*, with the budget-vs-rank
  phase transition as a named, characterized regime, and the variance-reduction benefit.
- **Why do this**: the pilot already shows a real, large, reliable effect in a well-defined regime.

### Idea 2: "The space matters" — gradient-space vs saliency-space low-rank asymmetry — BACKUP / fold-in
- **Hypothesis**: per-sample gradient matrices in *parameter* space are low-rank because all
  samples share one parameter basis; per-sample saliency matrices in *input* space are higher-rank,
  and the gap widens for text (token mismatch) vs images (shared pixel grid).
- **Pilot**: supported on images (7.6× normalized-rank gap). The text-token mechanism is **not yet
  tested** — needs a text pilot to confirm the strong version.
- **Contribution**: diagnostic. Publishable either way; explains why feature/saliency-space
  competitors (GRAFT-style) differ. Recommended as a core section of Idea 1, or a standalone note.
- **Risk**: LOW–MEDIUM. **Effort**: 1–2 weeks.

### Idea 3: Why loss-based selection fails at aggressive budgets — fold-in finding
- **Hypothesis (reframed from a falsified one)**: at aggressive budgets, neither easy- nor
  hard-sample selection spans the gradient subspace — easy-only collapses coverage, hard-only
  triggers the KD capacity-gap failure. Only geometric (subspace-spanning) selection works.
- **Pilot**: supported — easy_lowloss = 0.55 and hard_highloss = 0.95 (worst) at small budgets,
  vs D-optimal 0.91. This is the honest replacement for the falsified "structural ≠ hard" claim.
- **Contribution**: empirical finding; reconciles gradient-geometry selection with the KD
  capacity-gap literature. **Risk**: LOW. Fold into Idea 1.

### Idea 4: Coreset error bound with a budget-vs-rank phase transition — theory component
- **Hypothesis**: under a decaying gradient spectrum, a D-optimal coreset of size m yields a KD
  error bounded by the residual energy when m ≳ rank, and incurs an unavoidable
  high-variance coverage failure when m < rank — a provable phase transition matching the pilot.
- **Contribution**: theoretical result (likely tractable in the NTK/linearized-student regime).
- **Risk**: MEDIUM–HIGH. **Effort**: weeks. Recommended as a supporting section, not the lead.

---

## Other Ideas Considered (brainstorm breadth)

| # | Idea | Type | Verdict |
|---|------|------|---------|
| 5 | Re-select the coreset every T epochs as the gradient subspace drifts (a *curriculum* of coresets) — matches the project name | method | Promising extension of Idea 1; pilot later |
| 6 | Cheap-gradient ablation: last-layer / one-step / logit-difference proxies instead of full per-sample gradients | method | **Must-do** — determines whether the method nets a real compute saving |
| 7 | Structural ≠ hard: structural energy uncorrelated with loss | empirical | **Falsified by pilot** (corr +0.59) — reframed into Idea 3 |
| 1b | KD gradients *more* low-rank than CE | empirical | **Falsified by pilot** (wash) — drop the sub-claim |
| 8 | Coreset transferability across student seed / temperature / student size | empirical | Good cheap extension; fold into Idea 1 ablations |
| 9 | Operationalize Epiplexity: structural-info score = principal-subspace gradient energy; falsify via label-noise injection (noise should be residual-heavy) | empirical/framing | Keep as the framing layer + one decisive falsification experiment |
| 10 | Ridge-leverage-score sampling as a cheaper, provable alternative to greedy D-optimal | method | Minor variant / ablation of Idea 1 |
| 11 | "When does it fail" regime map (capacity gap, data heterogeneity) | diagnostic | Extension; depends on Idea 1 working |

## Eliminated / Demoted Ideas

| Idea | Reason |
|------|--------|
| "Save ~1/3 compute via modest pruning" (original framing) | Pilot: D-optimal = random at 50% budget. The method's value is at aggressive compression, not modest pruning. Re-anchored. |
| Idea 1b (KD more low-rank than CE) | Falsified — KD and CE gradient matrices have essentially equal effective rank. |
| Idea 7 as stated (structural ≠ hard) | Falsified — structural energy correlates +0.59 with loss. Reframed into Idea 3. |
| Standalone "apply DPP to gradients" | Subsumed by Idea 1's D-optimal criterion; not a separate contribution. |

## Suggested Execution Order

1. **Idea 1 (GradSpan-KD)** — scale the pilot to CIFAR-100 + a text KD pair; confirm the aggressive-
   budget win and variance reduction at real scale. Run the cheap-gradient ablation (Idea 6) in
   parallel — it gates the headline compute-saving claim.
2. **Idea 2 text pilot** — confirm the gradient-vs-saliency asymmetry on text (token-mismatch).
3. **Idea 4 (theory)** — formalize the budget-vs-rank phase transition once the empirics hold.
4. Ideas 5, 8, 9 as ablations/extensions inside the same paper.

## Phase 3: Deep Novelty Verification

**Verdict: PROCEED WITH CAUTION — overall novelty 5–6/10.** The method is more of a recombination
of known components than the original framing suggested. Each ingredient already exists; the
defensible novelty is narrower and must shift from "new algorithm" to "new finding + the right
space." (Cross-model check pending — Codex CLI unavailable; re-run `/novelty-check` when installed.)

### What is NOT novel (verified against full texts)
| Component | Owned by | Evidence |
|-----------|----------|----------|
| JL / random projection of per-sample gradients | LESS, TRAK, GraSS | LESS uses Rademacher JL projection (d=8192) of LoRA gradients. |
| D-optimal / volume-max / log-det selection criterion | **GRAFT**, DPP-MAP | GRAFT's "Fast MaxVol" maximizes submatrix \|det\| — **confirmed equivalent to D-optimal design**. The criterion itself is not new. |
| "SVD-extract a low-rank rep, then volume-max select" recipe | **GRAFT** | GRAFT's feature extractor `f` can be SVD; it then MaxVols. The *skeleton* (low-rank decompose → volume-select) already exists. |
| Task-agnostic gradient-based coreset selection | **TAGCOS** (arXiv 2407.15235) | TAGCOS clusters per-sample gradients and greedily selects a task-agnostic coreset (instruction tuning). |
| "Importance/loss-based selection collapses at high pruning; coverage is needed" | **CCS** (ICLR 2023) | CCS established the catastrophic-drop phenomenon and that coverage fixes it. The pilot's phase-transition phenomenon is qualitatively known. |
| KD + gradient-based data selection | **ICLR 2025** "Medium-Difficulty Samples…" | KD-specific dataset pruning using last-layer gradient L2-**norm** + difficulty. So "KD × gradient data selection" is **not empty** — but it uses gradient *magnitude*, not subspace geometry. |

### Closest prior work
| Paper | Overlap | Key difference |
|-------|---------|----------------|
| GRAFT (2508.13653) | low-rank decompose → MaxVol(=D-optimal) select | **feature** space, **per-batch** in-training, no Count Sketch, **no KD** |
| LESS (ICML 2024) | per-sample gradient → JL project → select | **target-similarity** ranking, no PCA/subspace, CE instruction tuning |
| TAGCOS (2407.15235) | task-agnostic per-sample gradient coreset | **clustering** of gradients (not PCA subspace / D-optimal), instruction tuning |
| CCS (ICLR 2023) | coverage beats importance at high pruning | coverage in **importance-score** space, not gradient subspace, not KD |
| ICLR 2025 Medium-Difficulty KD | KD data pruning via gradients | gradient **norm** + difficulty stratification, not gradient **geometry** |

### What survives as genuine novelty (the defensible core)
1. **The space asymmetry as an explanatory finding.** No prior work isolates that geometric data
   selection should be done in *parameter-gradient* space rather than feature/saliency/input space.
   This directly predicts that **GRAFT-style feature-space selection underperforms gradient-space
   selection** — a testable, novel, GRAFT-engaging claim. The pilot already shows
   `d_optimal_saliency` (0.967) < `d_optimal_grad` (0.971).
2. **The KD/KL-loss instantiation** + reconciliation with the KD capacity-gap literature
   (hard-sample selection is uniquely harmful in KD; geometric selection is not loss-based).
3. **Rank-calibrated budgeting** — tying the transition budget to the *measured* intrinsic gradient
   rank, not just "high pruning rate" (CCS does not calibrate to a measured rank).
4. **The Epiplexity formalization** — structural-info score = principal-subspace gradient energy;
   brand-new framework (Jan 2026), no prior connection.

### Required repositioning
- **Drop** the claim "we introduce gradient-subspace D-optimal coreset selection" — GRAFT + TAGCOS
  occupy that. **Drop** "KD × gradient data selection is unexplored" — ICLR 2025 exists.
- **Lead with the finding**, not the algorithm: *"For knowledge distillation, geometric data
  selection must be done in parameter-gradient space; doing it in feature/saliency space (à la
  GRAFT) or by loss magnitude (à la ICLR-2025-KD) measurably underperforms — and here is the
  D-optimal coreset that exploits the right space."* The method becomes the instantiation of the
  finding, not the headline.
- This makes Idea 2 (the space asymmetry) the **co-lead**, not a fold-in. Updated ranking below.

### Eliminated by Phase 3
- None outright. **Idea 1 downgraded** from "novel method" to "method instantiating a novel
  finding." **Idea 4 (phase transition)** demoted — CCS owns the qualitative phenomenon; only the
  rank-calibration angle is novel.

## Phase 4: External Critical Review

Senior-reviewer (NeurIPS/ICML-level) critique. **Done in-session (Claude Opus 4.7)** — the Codex
`gpt-5.4` backend was unavailable; re-run `/research-review` with Codex installed for an
independent cross-check.

### Mock review — scores
- **As a generic "new coreset method" paper: 4/10 (reject).** Incremental — GRAFT, TAGCOS, LESS,
  CCS collectively cover gradient/feature coreset selection and the high-pruning coverage story.
- **As a *finding* paper centered on the space asymmetry + KD capacity-gap hook, with strong
  experiments: 6/10 (borderline).** Publishable at a strong venue if the weaknesses below are
  closed; a solid workshop/second-tier paper otherwise.

### Weaknesses (ranked by severity)
1. **The "KD is special" story is currently broken.** The pilot falsified "KD gradients are more
   low-rank than CE" (they are equal). If the gradient geometry is not KD-specific, a reviewer asks
   "why is this a KD paper?" → **Fix:** make the KD hook the *capacity gap*, not the spectrum.
   The KD-specific value is that loss/difficulty-based selection is *uniquely harmful* in KD
   (hard samples drift the student decision boundary — ICLR 2025), whereas geometric selection is
   not loss-based and sidesteps this. Show a KD-vs-CE head-to-head where the method's *relative*
   advantage over loss-based selection is larger for KD.
2. **Method incrementality.** "GRAFT-style recipe in gradient space" reads as A+B. → **Fix:** lead
   with the finding ("the right space is parameter-gradient space"), run GRAFT and TAGCOS as
   first-class baselines (not strawmen). If feature-space GRAFT ties gradient-space selection, the
   paper collapses — this is the make-or-break experiment.
3. **The space-asymmetry comparison is confoundable.** Comparing rank across spaces of different
   ambient dimension invites attack. → **Fix:** compare *downstream coreset quality* under the
   *same* D-optimal criterion and *same* projected dimension across spaces (gradient / late-layer
   feature / input saliency); include text, where the token-mismatch mechanism is real (untested
   so far — the pilot was image-only, where pixels are shared coordinates).
4. **Selection cost vs savings not accounted.** Per-sample KD gradients cost ≈ one training epoch
   + one teacher forward per sample. Net saving is real only if (a) the coreset is then trained
   for many epochs and/or (b) teacher soft-label inference is the true bottleneck. → **Fix:**
   report **end-to-end wall-clock including selection**; run the cheap-gradient ablation (Idea 6) —
   it is mandatory, not optional. Motivate via **saving teacher inference / soft-label cost** and
   repeated distillation (many students, one teacher).
5. **Use-case realism.** "Distill on 2% of data" needs a real motivation. → **Fix:** frame as
   expensive-teacher soft-label budgets, data-scarce distillation, and reproducibility (variance
   reduction) — not "save 1/3 compute."
6. **Empirical-only.** Theory (Idea 4) is demoted. An empirical coreset paper needs breadth:
   ≥2 modalities, ≥3 teacher/student pairs, multiple budgets, strong baselines.

### Minimum viable improvement package (highest acceptance lift per GPU-week)
1. **Space head-to-head** (gradient vs late-layer feature vs input saliency), same criterion, on
   CIFAR-100 *and* a text KD pair. — the centerpiece.
2. **GRAFT + TAGCOS as native baselines.** — the make-or-break.
3. **KD-vs-CE relative-advantage experiment.** — restores the KD hook.
4. **End-to-end wall-clock + cheap-gradient ablation.** — defends the efficiency claim.
5. **Label-noise injection test** (Idea 9) — corrupted teacher logits should land in the residual
   subspace; cheap, decisive, and gives the Epiplexity framing teeth.

### Reviewer's bottom line
The pilot signal is real and the space-asymmetry finding is the genuine contribution. The project
is viable **if repositioned as a finding paper** and if GRAFT/TAGCOS are beaten head-to-head in
gradient space. If feature-space selection ties gradient-space selection at scale, **kill the
project** — that is the decisive go/no-go experiment and should be run first.

## Refined Proposal (Phase 4.5)

The method was refined and re-anchored. Full deliverables in `refine-logs/`:
- **Proposal**: `refine-logs/FINAL_PROPOSAL.md`
- **Experiment plan**: `refine-logs/EXPERIMENT_PLAN.md`
- **Experiment tracker**: `refine-logs/EXPERIMENT_TRACKER.md`
- **Pipeline summary**: `refine-logs/PIPELINE_SUMMARY.md`

**Problem anchor**: KD under an aggressive data/soft-label budget — which small subset to distill
on, and in what geometry to choose it.

**Final method thesis**: keep the subset whose per-sample KD-loss *parameter-gradients* span the
principal gradient subspace (D-optimal coverage); parameter-gradient space is the shared
coordinate system where samples' teacher–student-gap updates are low-rank and comparable.

**Dominant contribution (repositioned)**: a *finding* — geometric data selection for KD must be
done in **parameter-gradient space**; feature-space (GRAFT), input-saliency, and loss-magnitude
selection measurably underperform. **GradSpan-KD** is the instantiation, not the headline. This
elevates the original Idea 2 to **co-lead** with Idea 1.

**Must-run experiments**: 6 blocks (~14 GPU-days). **E1 (space head-to-head) is a hard go/no-go
gate** — run it before the rest.

**First 3 runs**: (1) E0 KD-gradient spectrum on ResNet-56→ResNet-20/CIFAR-100; (2) E1 vision
space head-to-head; (3) E1 text space head-to-head (token-mismatch test).

**Final verdict**: REVISE → conditionally proceed. Viable as a finding paper; novelty is 5–6/10,
so the repositioning and the E1 gate are essential.

## Next Steps
- [ ] `/run-experiment` — launch E0, then the E1 go/no-go gate (`refine-logs/EXPERIMENT_PLAN.md`).
- [ ] If E1 passes → run E2/E3/E4; if it ties on both modalities → pivot to a diagnostic note or kill.
- [ ] Install the Codex CLI (`claude mcp add codex -s user -- codex mcp-server`) and re-run
      `/novelty-check` and `/research-review` for an independent GPT-5.4 cross-check.
- [ ] After confirmed results → `/auto-review-loop` to iterate the paper toward submission.
