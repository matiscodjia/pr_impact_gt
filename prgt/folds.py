"""Out-of-fold assignment of each case, read from the nnU-Net result folders.

A case belongs to fold k if its prediction exists in ``fold_k/validation/`` of the M0
trainer. Reading the folders, rather than results/metrics.csv, keeps this usable on a node
that has the predictions but not the master table. Paths are relative to the repository
root.
"""
import glob
import os

RES = "nnUNet_data/nnUNet_results/Dataset100_PARSE"


def case_fold_map(trainer="nnUNetTrainerStd") -> dict:
    """{case file name: fold index} for every validation prediction of ``trainer``."""
    out = {}
    for d in sorted(glob.glob(f"{RES}/{trainer}__nnUNetPlans__3d_fullres/fold_*/validation")):
        fold = int(d.split("fold_")[1].split("/")[0])
        for f in glob.glob(os.path.join(d, "*.nii.gz")):
            out[os.path.basename(f)] = fold
    return out
