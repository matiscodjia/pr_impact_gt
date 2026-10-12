"""prgt: shared library for the PARSE reference-noise study.

Modules
-------
noise          The noise operator N(GT*; theta) that turns the clean reference GT* into a
               degraded reference GT- (distal omission, boundary drift, ...).
metrics        Per-case segmentation metrics (clDice, HD95, NSD, Betti-0 error, volume delta).
results_store  The long-format master table of per-case scores (results/metrics.csv).
stats          Paired statistics shared by the analyses (BCa bootstrap, Cohen's d, BH-FDR).
folds          Out-of-fold assignment of each case, read from the nnU-Net result folders.
"""
