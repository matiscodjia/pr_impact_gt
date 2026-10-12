# Experiment plan: effect of reference quality (PARSE, nnU-Net)

> Original design document of June 2026, translated in October 2026. It records the
> reasoning behind the models and metrics. Where it differs from the code, the code and
> `configs/experiment_config.yaml` are authoritative. In particular, the planned n = 15
> test cases became out-of-fold evaluation on all 85 cases, and the results are in
> `results/` (see `README.md`).

## Guiding principles

1. **One noise family at a time.** Simple, debuggable, clean causal attribution. No
   compound noise until each family is understood on its own.
2. **The instrument must match the quantity: in type, in scale AND in direction.**
   - **topological** noise (omission) -> **clDice, Betti-0**
   - **magnitude** of a boundary noise (drift, any mu) -> **NSD@0.5 mm**
     (NSD@2 mm is too coarse, blind to a ~1 mm drift; HD95 saturates below one voxel).
   - **direction** of a bias (drift mu != 0 = under/over-segmentation) -> **signed volume
     delta** (`compute_volume_delta`). NSD and HD95 are **unsigned**, so mu = -0.5 and
     mu = +0.5 give the same NSD. A directional bias needs a directional instrument.
   - **No clean dissociation on thin vessels** (checked, see step 0): a boundary shift of
     the order of the vessel radius **pinches** it, so the drift *really* affects the
     topology too (Betti-0). This is not an artefact to remove but the real phenomenon (an
     annotation error fragments the tree). For the drift, volume delta + NSD@0.5 + Betti-0
     are reported together; for the omission, clDice / Betti-0, with NSD staying flat as a
     control.
3. **Biased vs unbiased axis** (Heller 2018; Yao 2023 / GSD-Net):
   - **biased** noise (omission, drift mu != 0) penalises (Q1) **and** is learned (Q3);
   - **unbiased** noise (drift mu = 0) is almost invisible and acts as augmentation (Q2).

---

## Step 0: noise calibration against inter-annotator agreement (before any training)

`python pipeline/calibrate_noise.py` measures the agreement GT- vs GT* (no training) on a
few PARSE masks; output `results/noise_calibration.csv`. Measured results:

| Noise | Chosen parameters | Measured agreement (PARSE) | Status |
|---|---|---|---|
| `distal_omission` | **r=2, p=0.3** | clDice **0.878**, inside [0.85, 0.90] | kept |
| `boundary_drift` (r1 p0.5) | **r=1, p=0.5**, mu in [-0.5, +0.5] | NSD@0.5 **0.88**, inside [0.85, 0.90] | kept |
| `distal_truncation` | -- | (not relevant, see below) | **set aside (redundant)** |

The calibration was later remeasured on all 85 cases (`analysis/calibration_85.py`).

**Calibration findings (checked):**
- `distal_truncation` is **set aside because it is redundant with `distal_omission`**: same
  axis (biased reduction of the distal foreground). Its high clDice (0.97-0.99) does **not**
  show it is harmless: clDice only measures skeleton continuity and is **blind to boundary
  filing**. Wrong instrument; a surface metric would have been the right one. It is set
  aside for redundancy, not as a no-op.
- **Thin vessels: surface and topology cannot be separated (a real phenomenon, not an
  artefact).** Betti-0 uses 26-connectivity (N(GT*) = 1; 6-connectivity would give 99, a
  trap avoided). The drift *really* fragments the tree: N(degraded) ~ 59 at r1 p0.5
  (Betti-0 error 58), ~152 at r2 p1.0. A boundary error of the order of the vessel radius
  pinches it: a real annotation error, kept. Operating point **r1, p0.5** (the mildest that
  stays measurable).
- **HD95 saturates** (~0-0.9 mm, below one voxel), so it cannot grade the drift; NSD@0.5 is
  used instead.
- **NSD and HD95 are unsigned** and blind to the direction of mu. Q3 uses the **signed
  volume delta** (under-segmentation < 0 < over-segmentation; monotone in mu).

---

## Models to train

A shared baseline plus one model per condition (truncation left out for now).

| Model | Trained on | Regime | Serves |
|---|---|---|---|
| `M0_Star` | clean GT* | standard | baseline (Q1, Q2, Q3) |
| `M1_Omission` | distal_omission r2 p0.3 | on the fly | Q1, Q2 (topological axis) |
| `M2_Drift_mu0` | boundary_drift r1 p0.5 mu=0 | **on the fly** | Q2 (robustness) |
| `M3_Drift_muMinus` | boundary_drift r1 p0.5 mu=-0.5 (frozen) | **frozen** | Q3 (learned under-segmentation bias) |
| `M4_Drift_muPlus` | boundary_drift r1 p0.5 mu=+0.5 (frozen) | **frozen** | Q3 (learned over-segmentation bias) |

**Why the regime differs** (by design):
- **on the fly** (noise redrawn every epoch) for Q2: the augmentation IS the resampling. A
  frozen mu = 0 would learn the frozen ripple (worse), not robustness.
- **frozen** (one realisation per case) for Q3: one annotator cohort = one consistent set;
  a stable target makes the bias learnable, without confusing it with augmentation. This is
  the real-world scenario.

---

## Q1: does GT- artificially penalise a model?

**Set-up.** `M0_Star` (never degraded) evaluated on **GT\* then on GT-**: only the
evaluation reference changes. Paired Wilcoxon.

| Evaluation GT- | Matched metric | Expected |
|---|---|---|
| omission (r2 p0.3) | clDice down, Betti-0 error up; NSD flat (control) | **significant drop** |
| drift mu != 0 (r1 p0.5) | NSD@0.5 down, Betti-0 error up, volume delta != 0 (coupled) | **significant drop** |

**Expected result.** The same model loses points as soon as the evaluation reference
changes: the penalty comes from the **quality of the reference, not from the model**. For
the omission, a flat NSD shows the instrument is right; for the drift, surface AND topology
move (a real coupling on thin structures, not a defect).
**Null**: the degraded reference does not penalise (unlikely on the topological side).

## Q2: does unbiased noise act as augmentation (robustness)?

**Set-up.** `M2_Drift_mu0` (on the fly) vs `M0_Star`, **both evaluated on the clean GT\***,
NSD@0.5 + Betti-0 (the mu = 0 drift also fragments, so both are followed). mu = 0 isolates
robustness (no directional bias to match, volume delta ~ 0, which rules out Q3).

**Expected result.** `M2 >= M0` on clean data: at least **no harm** (the zero-mean noise
averages out, so the true boundary is learned), at best a small regularisation benefit.
**Bonus** (one more model): frozen mu = 0 < on-the-fly mu = 0 would show that resampling is
what brings robustness.
**Likely outcome: a tiny difference, below the detection threshold.** "No effect" and
"effect below the metric's sensitivity" could then not be told apart, which **motivates the
metrology** (minimum detectable difference, metric sensitivity; link with
`metric_reliability_study`).

## Q3: can a segmentation bias be learned?

**Set-up.** `M3_Drift_muMinus` trained on a **frozen** GT- (mu < 0, a consistent bias per
case), evaluated against two references. **Signed** volume delta (directional) + NSD@0.5.
Paired Wilcoxon on the penalties. (The final, pre-registered analysis plan is in
`analysis/q3_learned_bias.py`; its primary contrast is M4 vs M3, not Mk vs M0.)

| Evaluated against | Expected for M3 | vs M0_Star |
|---|---|---|
| **biased GT-** (the "cohort") | **small** penalty (bias learned); volume delta(M3) ~ volume delta(GT-) | M0 penalised **more** |
| **true GT\*** | **worse** than M0; volume delta(M3) < 0 (under-segments as learned) | -- |

**Expected result.** The model trained on the bias **fits the wrong data better** (smaller
penalty on the biased GT-) **and** moves away from the truth (worse on GT\*): the bias is
internalised. Mirror image for `M4_Drift_muPlus`.
**Null**: the same penalty for M3 and M0 (unlikely: mu is a constant, learnable signal).

---

## Statistics (common to all questions)

- **Test**: paired Wilcoxon signed-rank, no normality assumption.
- **Effect size**: paired Cohen's d.
- **If not significant**: report the **minimum detectable difference** (the smallest
  difference detectable given the between-case variance) and do NOT conclude "no effect".
  This is the bridge to metrology.
- **Pre-registration**: the matched metric is the primary endpoint of each family. For the
  omission, NSD is the control (dissociation). For the drift, surface and topology are
  coupled (no flat control), so the primary endpoint is the signed volume delta (direction).

## Expected overall story

Two levels of "reaction" to tell apart: (i) **noise <-> metric** (GT- vs GT*);
(ii) **model <-> model** (trained models, on held-out cases).

| Noise | (i) distorts the reference? | (ii) separates the models? | Conclusion |
|---|---|---|---|
| omission, drift mu != 0 (**biased**) | yes (matched metric) | yes | penalises (Q1) **and** is learned (Q3) |
| drift mu = 0 (**unbiased**) | yes (NSD, Betti-0 at r1 p0.5) | **probably not / below the threshold** | benign, acts as augmentation (Q2), **motivates the metrology** |

**Guideline: biased noise changes the ranking and is learned.** Unbiased noise does distort
one realisation of the reference, but its effect on the *model* (on the fly, averaged) will
be small, possibly below the sensitivity of the metrics. This is not a failure: it is what
justifies studying the **sensitivity of the metrics themselves**, and links back to
`metric_reliability_study`.
