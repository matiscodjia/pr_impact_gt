"""Pli OOF de chaque cas, lu directement dans les dossiers de prédictions nnU-Net.

Évite de dépendre de results/metrics.csv (non suivi par git, donc absent sur le cluster) :
un cas appartient au pli k si sa prédiction existe dans fold_k/validation/ du trainer M0.
"""
import glob
import os

RES = "nnUNet_data/nnUNet_results/Dataset100_PARSE"


def case_fold_map(trainer="nnUNetTrainerStd") -> dict:
    out = {}
    for d in sorted(glob.glob(f"{RES}/{trainer}__nnUNetPlans__3d_fullres/fold_*/validation")):
        fold = int(d.split("fold_")[1].split("/")[0])
        for f in glob.glob(os.path.join(d, "*.nii.gz")):
            out[os.path.basename(f)] = fold
    return out
