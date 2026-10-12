#!/usr/bin/env python3
"""Génère les figures 2D individuelles (une par famille, une par μ) en un seul
processus : le masque est chargé une fois, puis chaque figure est rendue et
sauvegardée. Évite le churn de processus et la pression mémoire des appels répétés.

    python scripts/gen_individual_figs.py [--case PARSE_0001] [--out report/figures]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualize_degradations import (_load_mask, render_2d, render_mu_sweep_2d,
                                     render_3d, render_mu_sweep_3d, set_theme)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="PARSE_0001")
    ap.add_argument("--labels", default="nnUNet_data/nnUNet_raw/Dataset100_PARSE/labelsTr")
    ap.add_argument("--out", default="report/figures")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--with-3d", action="store_true", help="génère aussi les rendus 3D individuels")
    args = ap.parse_args()

    set_theme(True)  # fond clair (paper)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.labels, f"{args.case}.nii.gz")
    original, spacing = _load_mask(path)
    print(f"[{args.case}] shape={original.shape} fg={int(original.sum())}")

    # Chaque famille à son point d'opération calibré :
    #   omission r2 p0.3 (clDice 0.87) ; drift r1 p0.5 (NSD@0.5 0.88) ;
    #   contrôles à titre illustratif (r2 p0.5).
    FAM_RP = {"distal_omission": (2.0, 0.3), "boundary_drift": (1.0, 0.5),
              "distal_truncation": (2.0, 0.5), "homogeneous_morpho": (2.0, 0.5)}
    for fam, (r, p) in FAM_RP.items():
        out = os.path.join(args.out, f"ind_2d_{fam}.png")
        render_2d(original, spacing, [fam], r, p, 0, out, args.case, dpi=args.dpi, mu=0.0)
        ok = os.path.getsize(out) if os.path.exists(out) else 0
        print(f"  {'✓' if ok else '✗'} {os.path.basename(out)}  {ok//1024} Ko")

    for name, mu in (("neg", -0.5), ("zero", 0.0), ("pos", 0.5)):
        out = os.path.join(args.out, f"ind_2d_mu_{name}.png")
        render_mu_sweep_2d(original, spacing, 1.0, 0.5, [mu], 0, out, args.case, dpi=args.dpi)
        ok = os.path.getsize(out) if os.path.exists(out) else 0
        print(f"  {'✓' if ok else '✗'} {os.path.basename(out)}  (μ={mu})  {ok//1024} Ko")

    if args.with_3d:
        print("--- 3D (plus lent) ---")
        for fam in ("distal_omission", "boundary_drift"):
            r, p = FAM_RP[fam]
            out = os.path.join(args.out, f"ind_3d_{fam}.png")
            render_3d(original, spacing, [fam], r, p, 0, out, args.case, 2, dpi=args.dpi, mu=0.0)
            ok = os.path.getsize(out) if os.path.exists(out) else 0
            print(f"  {'✓' if ok else '✗'} {os.path.basename(out)}  {ok//1024} Ko")
        for name, mu in (("neg", -0.5), ("zero", 0.0), ("pos", 0.5)):
            out = os.path.join(args.out, f"ind_3d_mu_{name}.png")
            render_mu_sweep_3d(original, spacing, 1.0, 0.5, [mu], 0, out, args.case, 2, dpi=args.dpi)
            ok = os.path.getsize(out) if os.path.exists(out) else 0
            print(f"  {'✓' if ok else '✗'} {os.path.basename(out)}  (μ={mu})  {ok//1024} Ko")


if __name__ == "__main__":
    main()
