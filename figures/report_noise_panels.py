#!/usr/bin/env python3
"""Individual 2D (and optionally 3D) noise panels of the technical report, one per family
and one per drift bias mu.

The mask is loaded once and every figure is rendered in the same process (repeated calls
of noise_views.py would reload it each time).

    python figures/report_noise_panels.py [--case PARSE_0001] [--out report/technical/figures] [--with-3d]
"""
import argparse
import os

from noise_views import load_mask, render_2d, render_3d, render_mu_sweep_2d, render_mu_sweep_3d, set_theme


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="PARSE_0001")
    ap.add_argument("--labels", default="nnUNet_data/nnUNet_raw/Dataset100_PARSE/labelsTr")
    ap.add_argument("--out", default="report/technical/figures")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--with-3d", action="store_true", help="also render the individual 3D views")
    args = ap.parse_args()

    set_theme(True)  # light background (paper)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.labels, f"{args.case}.nii.gz")
    original, spacing = load_mask(path)
    print(f"[{args.case}] shape={original.shape} fg={int(original.sum())}")

    # Each family at its calibrated operating point:
    #   omission r2 p0.3 (clDice 0.87); drift r1 p0.5 (NSD@0.5 0.88);
    #   the two controls for illustration only (r2 p0.5).
    FAM_RP = {"distal_omission": (2.0, 0.3), "boundary_drift": (1.0, 0.5),
              "distal_truncation": (2.0, 0.5), "homogeneous_morpho": (2.0, 0.5)}
    for fam, (r, p) in FAM_RP.items():
        out = os.path.join(args.out, f"ind_2d_{fam}.png")
        render_2d(original, spacing, [fam], r, p, 0, out, args.case, dpi=args.dpi, mu=0.0)
        ok = os.path.getsize(out) if os.path.exists(out) else 0
        print(f"  {'ok' if ok else 'MISSING'} {os.path.basename(out)}  {ok//1024} KB")

    for name, mu in (("neg", -0.5), ("zero", 0.0), ("pos", 0.5)):
        out = os.path.join(args.out, f"ind_2d_mu_{name}.png")
        render_mu_sweep_2d(original, spacing, 1.0, 0.5, [mu], 0, out, args.case, dpi=args.dpi)
        ok = os.path.getsize(out) if os.path.exists(out) else 0
        print(f"  {'ok' if ok else 'MISSING'} {os.path.basename(out)}  (mu={mu})  {ok//1024} KB")

    if args.with_3d:
        print("--- 3D (slower) ---")
        for fam in ("distal_omission", "boundary_drift"):
            r, p = FAM_RP[fam]
            out = os.path.join(args.out, f"ind_3d_{fam}.png")
            render_3d(original, spacing, [fam], r, p, 0, out, args.case, 2, dpi=args.dpi, mu=0.0)
            ok = os.path.getsize(out) if os.path.exists(out) else 0
            print(f"  {'ok' if ok else 'MISSING'} {os.path.basename(out)}  {ok//1024} KB")
        for name, mu in (("neg", -0.5), ("zero", 0.0), ("pos", 0.5)):
            out = os.path.join(args.out, f"ind_3d_mu_{name}.png")
            render_mu_sweep_3d(original, spacing, 1.0, 0.5, [mu], 0, out, args.case, 2, dpi=args.dpi)
            ok = os.path.getsize(out) if os.path.exists(out) else 0
            print(f"  {'ok' if ok else 'MISSING'} {os.path.basename(out)}  (mu={mu})  {ok//1024} KB")


if __name__ == "__main__":
    main()
