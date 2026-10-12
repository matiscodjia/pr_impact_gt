# Reference noise and segmentation-metric rankings (PARSE, nnU-Net)

When the reference annotation of a test set is itself noisy, can it change which model a
segmentation metric ranks first? This repository studies the question on pulmonary-artery
segmentation in CT (PARSE, 85 cases) with nnU-Net.

- A **noise operator** N(GT*; theta) turns the clean reference GT* into degraded references
  GT- (omission of thin distal branches, biased or unbiased boundary drift), calibrated so
  that the agreement between GT- and GT* stays in an inter-observer range
  (clDice 0.85-0.90).
- Three nnU-Net models are trained on the same 85 cases with 5-fold cross-validation:
  **M0** on clean labels, **M1** with on-the-fly omission noise, **M2** with on-the-fly
  unbiased boundary noise. Every case is scored out of fold against GT* and against each
  GT-.
- The analyses test whether the **ranking** of the models changes with the reference
  (interaction test, `analysis/rank_reversal.py`), and how robust that change is (noise
  dose, noise draws, structured omission, post-processing, folds, training seeds).
- Q3 asks whether a network trained on a biased reference learns the bias (models M3/M4 on
  frozen biased labels; analysis plan fixed before the data in
  `analysis/q3_learned_bias.py`).

## Repository layout

```
prgt/              shared library (installed with pip install -e .)
  noise.py           noise operator N(GT*; theta): distal_omission, boundary_drift, ...
  metrics.py         clDice, HD95, NSD@2/@0.5, Betti-0 error, signed volume delta
  results_store.py   master table of per-case scores (results/metrics.csv)
  stats.py           paired statistics: BCa bootstrap, Cohen's d, Wilcoxon, BH-FDR
  folds.py           out-of-fold assignment of each case
pipeline/          data -> training -> scores
  convert_parse.py             PARSE -> nnU-Net raw format (Dataset100)
  make_eval_references.py      degraded evaluation references GT- (labelsTr_GT_minus_*)
  make_fixed_training_sets.py  frozen degraded training labels (Dataset103/104)
  orchestrator.py              resumable k-fold training, then scoring and report
  collect_metrics[_parallel].py  scores of every prediction against every reference
  report.py, report_stats.py, report_figures.py   running report (results/STATUS.md)
  calibrate_noise.py           noise calibration (agreement GT- vs GT*)
  calibrate_epochs.py          choice of num_epochs (topology convergence)
  run_pipeline.sh              the whole chain, step by step
nnunet_trainers/   custom nnU-Net trainers (on-the-fly noise, seed replicates, calibration)
analysis/          one script per result of the paper (reads results/, writes results/)
figures/           figures of the technical report (noise views, convergence)
cluster/           GPU-node scripts (launcher, progress, Q3 preparation); see cluster/README.md
configs/           experiment_config.yaml: models, noise, references, training
results/           master table, analysis outputs, figures, cluster_runs/ (GPU-node scores)
report/technical/  technical report (LaTeX, in French)
docs/              experiment_plan.md: original design rationale
tests/             unit tests of prgt.stats
```

## Installation

```bash
bash setup_env.sh --local        # or --cluster on the GPU node: venv, PyTorch, nnU-Net, prgt
source .env_nnunet               # at every session: nnU-Net paths + venv
```

For the analyses only (no training), `pip install -e .` in a Python 3.11 environment is
enough. All scripts are run **from the repository root** (paths such as
`nnUNet_data/...` and `results/...` are relative to it).

The PARSE data are not included. `pipeline/convert_parse.py` builds
`nnUNet_data/nnUNet_raw/Dataset100_PARSE` from a local copy of the dataset.

## Reproducing the study

### 1. Data, references and training (GPU)

```bash
bash pipeline/run_pipeline.sh /path/to/PARSE            # M0-M2 (tier A), 5 folds
bash pipeline/run_pipeline.sh /path/to/PARSE --tiers AB --from q3_datasets   # also M3/M4
```

Steps: conversion -> nnU-Net preprocessing -> degraded references (seed 42 + case index)
-> frozen training sets of M3/M4 (with the plans of Dataset100) -> resumable training.
After each fold, `pipeline/collect_metrics.py` adds the new scores to `results/metrics.csv`
(one row per model x fold x reference x case, with provenance). The study's table was
computed in July 2026 (commit c1e84f2). Seed replicates and Q3 ran on a shared GPU node with
the scripts of `cluster/`.

### 2. Analyses (CPU)

Each script documents its question, inputs and outputs in its docstring.

**Fast** (seconds; read the CSV files of `results/`):

| Result | Script | Output in `results/` |
|---|---|---|
| Ranking reversal (interaction test, FDR over 126 tests) | `analysis/rank_reversal.py` | `rank_reversal_tests.csv`, `inter_model_by_reference.csv`, `rank_matrix.csv`, `rank_stability_bootstrap.csv` |
| Robustness checks (signed volume, MDD, Wilcoxon floor) | `analysis/reversal_robustness.py` | `reversal_robustness_checks.csv` |
| MDD rule applied to every reversal | `analysis/mdd_crossing.py` | `mdd_crossing.csv` |
| Reference effect vs model effect | `analysis/effect_ratio.py` | `effect_ratio.csv` |
| Fold-by-fold stability | `analysis/fold_stability.py` | `fold_stability.csv` |
| Training-seed variance | `analysis/seed_variance.py` | `seed_variance.csv` |
| Spread over independent GT- draws | `analysis/reference_draws.py` | `reference_draws_summary.csv` |
| HD95 mechanism (omitted volume vs penalty) | `analysis/mechanism_correlation.py` | `mechanism_correlation.csv` |
| LCC post-processing | `analysis/lcc_reversal.py` | `lcc_reversal_check.csv`, `lcc_cldice_cost.csv` |
| Q3: learned bias (M3/M4) | `analysis/q3_learned_bias.py` | `q3_learned_bias.csv` |
| Figure 1 of the paper | `analysis/figure1_isbi.py` | `figures/fig1_isbi.pdf` |

**Heavy** (minutes to hours; read the NIfTI volumes; most have `--stats-only` to rerun
the statistics from their per-case CSV, and `--limit N` for a trial on N cases):

| Result | Script | Output in `results/` |
|---|---|---|
| Severity of each reference vs GT* | `analysis/reference_severity.py` | `reference_severity.csv` |
| Noise calibration on all 85 cases | `analysis/calibration_85.py` | `calibration_85_*.csv` |
| Dose-response in the omission rate p | `analysis/dose_response.py` | `dose_response*.csv` |
| Structured (generation / territory) omission | `analysis/structured_omission.py` | `structured_omission*.csv` |
| Scores with largest-component post-processing | `analysis/lcc_metrics.py` | `metrics_lcc.csv` |
| Coverage of the omitted regions | `analysis/omitted_coverage.py` | `omitted_coverage_*.csv` |
| Connected components behind Betti-0 | `analysis/betti0_counts.py` | `betti0_counts.csv` |
| Components of GT* by connectivity | `analysis/gtstar_connectivity.py` | `gtstar_connectivity.csv` |

Order where it matters: `reference_severity` before `mechanism_correlation`; `lcc_metrics`
before `lcc_reversal`; `calibration_85` before `figure1_isbi`; the dose-response runs
(`--tag _fine`, `--tag _p05`, the draws of `cluster/launch_cpu_draws.sh`),
`structured_omission` and `lcc_metrics` before `mdd_crossing`.

### 3. Tests

```bash
pip install -e ".[dev]" && pytest
```

## Models and references

| Model | Trainer | Training labels |
|---|---|---|
| M0_Star | `nnUNetTrainerStd` | clean GT* |
| M1_Omission | `nnUNetTrainerDegradedOmissionOnly` | distal_omission r2 p0.3, redrawn each epoch |
| M2_Drift_mu0 | `nnUNetTrainerDriftMu0` | boundary_drift r1 p0.5 mu=0, redrawn each epoch |
| M3_Drift_muMinus | `nnUNetTrainerStd` on Dataset103 | frozen boundary_drift r1 p0.5 mu=-0.5 |
| M4_Drift_muPlus | `nnUNetTrainerStd` on Dataset104 | frozen boundary_drift r1 p0.5 mu=+0.5 |
| `*_s1`, `*_s2` | seed replicates of M0-M2 | as the base model, another seed |

| Reference | Definition |
|---|---|
| GT_star | clean PARSE label |
| GT_minus_omission | distal_omission r2 p0.3 (clDice to GT* 0.878) |
| GT_minus_drift_neg | boundary_drift r1 p0.5 mu=-0.5 (under-segmenting annotator) |
| GT_minus_drift_pos | boundary_drift r1 p0.5 mu=+0.5 (over-segmenting annotator) |

Metric columns of `results/metrics.csv`: `cldice`, `hd95` (mm), `nsd` (NSD at 2 mm),
`nsd05` (NSD at 0.5 mm), `betti0` (|components(pred) - components(ref)|, 26-connectivity),
`volume_delta` ((|pred| - |ref|) / |ref|, signed).
