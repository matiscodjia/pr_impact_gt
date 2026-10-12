#!/usr/bin/env python3
"""Does the scoring on the GPU node reproduce the study's scores exactly?

The seed-variance analysis compares the study's scores (seed 0, results/metrics.csv,
computed in July 2026 at commit c1e84f2) with those of the replicates, computed on the GPU
node. What can differ is the environment (numpy, scipy, scikit-image, nibabel versions).
This script rescores 3 study cases (M0, fold 0: lowest, median and highest HD95 under
omission) against the 4 references through the code path of the pipeline
(prgt.metrics.PredictionFeatures + score_prediction) and compares with the 72 original
values, copied below at full precision.

Run ON the node, from the repository root, once the GT- references are there:
    python3 cluster/check_scoring.py          # ~10-30 min of CPU depending on the load
Output: "identical" or the differences per row, then a verdict (exit code 1 if any differs).
"""
import os
import sys
import time

import nibabel as nib

from prgt.metrics import PredictionFeatures, score_prediction

RAW = os.environ.get("nnUNet_raw", "nnUNet_data/nnUNet_raw") + "/Dataset100_PARSE"
RES = os.environ.get("nnUNet_results", "nnUNet_data/nnUNet_results") + "/Dataset100_PARSE"
PRED = RES + "/nnUNetTrainerStd__nnUNetPlans__3d_fullres/fold_0/validation"
REF_DIR = {"GT_star": "labelsTr", "GT_minus_omission": "labelsTr_GT_minus_omission",
           "GT_minus_drift_neg": "labelsTr_GT_minus_drift_neg",
           "GT_minus_drift_pos": "labelsTr_GT_minus_drift_pos"}
TOL = 1e-9          # relative: beyond this it is no longer rounding

# Values of results/metrics.csv (M0_Star, fold 0, commit c1e84f2, computed on 2026-07-12)
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
            print(f"{case}: study prediction missing ({pred_path})")
            n_skip += 4
            continue
        nii = nib.load(pred_path)
        pf = PredictionFeatures(nii.get_fdata(), spacing=nii.header.get_zooms()[:3])
        for scen, ref in REF_DIR.items():
            gt_path = os.path.join(RAW, ref, case)
            if not os.path.exists(gt_path):
                print(f"{case} {scen:<19} reference missing ({ref}): copy the GT- references first")
                n_skip += 1
                continue
            got = score_prediction(pf, nib.load(gt_path).get_fdata())
            want = EXPECTED[(case, scen)]
            bad = {m: (want[m], got[m]) for m in want
                   if abs(float(got[m]) - want[m]) > TOL * max(1.0, abs(want[m]))}
            if bad:
                n_diff += 1
                print(f"{case} {scen:<19} DIFFERENCE: " +
                      ", ".join(f"{m} {w:.6g} -> {g:.6g}" for m, (w, g) in bad.items()))
            else:
                n_ok += 1
                print(f"{case} {scen:<19} identical (6 metrics)")
        print(f"   ({time.time() - t0:.0f} s)")
    print()
    if n_diff:
        print(f"VERDICT: {n_diff} row(s) differ out of {n_ok + n_diff} -- the node's scores are NOT "
              "interchangeable with the study's; do not mix seed 0 and the replicates without rescoring.")
    elif n_skip:
        print(f"VERDICT: {n_ok} rows identical, {n_skip} not checked (missing files).")
    else:
        print(f"VERDICT: {n_ok}/{n_ok} rows identical to the study -- the node's scores are comparable.")
    sys.exit(1 if n_diff else 0)


if __name__ == "__main__":
    main()
