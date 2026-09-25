#!/usr/bin/env python3
"""Le calcul des scores du CLUSTER redonne-t-il exactement les scores de l'étude ?

La variance de graine compare les scores de l'étude (graine 0, results/metrics.csv, calculés
en juillet au commit c1e84f2) à ceux des réplicats, calculés sur le cluster. Le code de
scoring n'a pas changé depuis (aucun commit sur cross_evaluate.py, collect_metrics.py,
results_store.py, degradations.py) ; reste l'environnement (versions de numpy, scipy,
scikit-image, nibabel). Ce script rescore 3 cas de l'étude (M0, pli 0 : HD95 sous omission
le plus bas, médian, le plus haut) contre les 4 références, par le MÊME chemin de code que
l'orchestrateur (PredFeatures + evaluate_pair_cached), et compare aux 72 valeurs d'origine,
recopiées ci-dessous en pleine précision.

À lancer SUR le cluster, racine du repo, après la copie des références GT⁻ :
    python3 scripts/cluster/check_scoring.py          # ~10-30 min de CPU selon la charge
Sortie : « identique » par ligne ou les écarts, puis un verdict.
"""
import os
import sys
import time

import nibabel as nib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cross_evaluate import PredFeatures, evaluate_pair_cached  # noqa: E402

RAW = os.environ.get("nnUNet_raw", "nnUNet_data/nnUNet_raw") + "/Dataset100_PARSE"
RES = os.environ.get("nnUNet_results", "nnUNet_data/nnUNet_results") + "/Dataset100_PARSE"
PRED = RES + "/nnUNetTrainerStd__nnUNetPlans__3d_fullres/fold_0/validation"
REF_DIR = {"GT_star": "labelsTr", "GT_minus_omission": "labelsTr_GT_minus_omission",
           "GT_minus_drift_neg": "labelsTr_GT_minus_drift_neg",
           "GT_minus_drift_pos": "labelsTr_GT_minus_drift_pos"}
TOL = 1e-9          # relatif : au-delà, ce n'est plus de l'arrondi

# Valeurs de results/metrics.csv (M0_Star, pli 0, commit c1e84f2, calcul du 2026-07-12)
EXPECTED = {
    ("PARSE_0007.nii.gz", "GT_star"): {"cldice": 0.8778773898446248, "hd95": 0.6796875, "nsd": 0.958821394397334, "nsd05": 0.7277444154815719, "betti0": 43.0, "volume_delta": -0.0494327721096527},
    ("PARSE_0007.nii.gz", "GT_minus_omission"): {"cldice": 0.8188084085808764, "hd95": 0.6796875, "nsd": 0.9375213103433392, "nsd05": 0.7038908910421028, "betti0": 32.0, "volume_delta": -0.0092032689671296},
    ("PARSE_0007.nii.gz", "GT_minus_drift_neg"): {"cldice": 0.8490922516897292, "hd95": 0.6796875, "nsd": 0.9540882830304508, "nsd05": 0.6722898253344882, "betti0": 30.0, "volume_delta": -0.0099748328383673},
    ("PARSE_0007.nii.gz", "GT_minus_drift_pos"): {"cldice": 0.8669427342645579, "hd95": 0.961223280675463, "nsd": 0.95231758531211, "nsd05": 0.6471704833211351, "betti0": 29.0, "volume_delta": -0.1044981125792828},
    ("PARSE_0027.nii.gz", "GT_star"): {"cldice": 0.9162936752668952, "hd95": 1.0, "nsd": 0.9566217969071124, "nsd05": 0.550280959026501, "betti0": 87.0, "volume_delta": -0.217446471381944},
    ("PARSE_0027.nii.gz", "GT_minus_omission"): {"cldice": 0.8254449673596372, "hd95": 1.3201044584615265, "nsd": 0.9292880509320284, "nsd05": 0.5330625923890131, "betti0": 55.0, "volume_delta": -0.1562114170924922},
    ("PARSE_0027.nii.gz", "GT_minus_drift_neg"): {"cldice": 0.8943218543931444, "hd95": 1.1710413701594833, "nsd": 0.9571829531563588, "nsd05": 0.5535187776970716, "betti0": 11.0, "volume_delta": -0.1791770541186247},
    ("PARSE_0027.nii.gz", "GT_minus_drift_pos"): {"cldice": 0.9017817465300694, "hd95": 1.21875, "nsd": 0.953659966695593, "nsd05": 0.4858315547529141, "betti0": 68.0, "volume_delta": -0.2631522712570348},
    ("PARSE_0005.nii.gz", "GT_star"): {"cldice": 0.9216992697617398, "hd95": 4.038178396953879, "nsd": 0.9526378575173056, "nsd05": 0.7409034058309014, "betti0": 66.0, "volume_delta": 0.2002587715214099},
    ("PARSE_0005.nii.gz", "GT_minus_omission"): {"cldice": 0.7989856271018864, "hd95": 9.388253164745375, "nsd": 0.8878168569933308, "nsd05": 0.6613795663862088, "betti0": 54.0, "volume_delta": 0.3297027875207679},
    ("PARSE_0005.nii.gz", "GT_minus_drift_neg"): {"cldice": 0.8832070330749271, "hd95": 4.518518279955174, "nsd": 0.9459295546662452, "nsd05": 0.6604111233070954, "betti0": 42.0, "volume_delta": 0.2811161155662968},
    ("PARSE_0005.nii.gz", "GT_minus_drift_pos"): {"cldice": 0.90902947097254, "hd95": 3.523777439948366, "nsd": 0.95451328268768, "nsd05": 0.6805313075071958, "betti0": 46.0, "volume_delta": 0.1097758262075341},
}


def main():
    import numpy, scipy, skimage
    print(f"numpy {numpy.__version__} | scipy {scipy.__version__} | scikit-image "
          f"{skimage.__version__} | nibabel {nib.__version__}")
    n_ok = n_diff = n_skip = 0
    for case in dict.fromkeys(c for c, _ in EXPECTED):
        t0 = time.time()
        pred_path = os.path.join(PRED, case)
        if not os.path.exists(pred_path):
            print(f"{case} : prédiction de l'étude absente ({pred_path})")
            n_skip += 4
            continue
        nii = nib.load(pred_path)
        pf = PredFeatures(nii.get_fdata(), spacing=nii.header.get_zooms()[:3])
        for scen, ref in REF_DIR.items():
            gt_path = os.path.join(RAW, ref, case)
            if not os.path.exists(gt_path):
                print(f"{case} {scen:<19} référence absente ({ref}) -- copier les GT⁻ depuis le Mac")
                n_skip += 1
                continue
            got = evaluate_pair_cached(pf, nib.load(gt_path).get_fdata())
            want = EXPECTED[(case, scen)]
            bad = {m: (want[m], got[m]) for m in want
                   if abs(float(got[m]) - want[m]) > TOL * max(1.0, abs(want[m]))}
            if bad:
                n_diff += 1
                print(f"{case} {scen:<19} ÉCART : " +
                      ", ".join(f"{m} {w:.6g} -> {g:.6g}" for m, (w, g) in bad.items()))
            else:
                n_ok += 1
                print(f"{case} {scen:<19} identique (6 métriques)")
        print(f"   ({time.time() - t0:.0f} s)")
    print()
    if n_diff:
        print(f"VERDICT : {n_diff} ligne(s) différente(s) sur {n_ok + n_diff} -- les scores du cluster ne sont "
              "PAS interchangeables avec ceux de l'étude ; ne pas mélanger graine 0 et réplicats sans rescorer.")
    elif n_skip:
        print(f"VERDICT : {n_ok} lignes identiques, {n_skip} non vérifiées (fichiers absents).")
    else:
        print(f"VERDICT : {n_ok}/{n_ok} lignes identiques à l'étude -- scores du cluster et de juillet comparables.")
    sys.exit(1 if n_diff else 0)


if __name__ == "__main__":
    main()
