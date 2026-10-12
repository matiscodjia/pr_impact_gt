#!/usr/bin/env python3
"""Spread of the result over independent draws of GT- (variance of the noise realisation).

Inputs : results/dose_response.csv (the study's draw, seed 42, row p = 0.3) and
         results/dose_response_draw*.csv (one file per seed, see dose_response.py --tag).
Output : results/reference_draws_summary.csv + printed table.

For each (metric, pair): mean interaction per draw, then mean / standard deviation / min /
max over draws, and the number of draws in which the sign of the M0 - Mk gap differs
between GT* and GT-. All draws use the SAME 85 cases, so they are not independent: the
spread of the means is reported, not a confidence interval.

    python analysis/reference_draws.py
"""
import glob
import os
import re

import pandas as pd

P = 0.3


def main():
    frames = []
    base = "results/dose_response.csv"
    if os.path.exists(base):
        frames.append(pd.read_csv(base).assign(draw="42"))
    for f in sorted(glob.glob("results/dose_response_draw*.csv")):
        if re.search(r"_(metrics|checks)\.csv$", f):
            continue
        frames.append(pd.read_csv(f).assign(draw=re.search(r"draw(\d+)", f).group(1)))
    df = pd.concat(frames, ignore_index=True)
    df = df[(df.p_omission - P).abs() < 1e-9]
    rows = []
    for (metric, pair), g in df.groupby(["metric", "pair"]):
        rows.append({"metric": metric, "pair": pair, "n_draws": g.draw.nunique(),
                     "interaction_mean_over_draws": g.interaction_mean.mean(),
                     "interaction_sd_over_draws": g.interaction_mean.std(ddof=1),
                     "interaction_min": g.interaction_mean.min(), "interaction_max": g.interaction_mean.max(),
                     "delta_ref_mean": g.delta_ref.mean(), "delta_ref_sd": g.delta_ref.std(ddof=1),
                     "n_draws_sign_flip": int(g.sign_flip.sum()),
                     "d_min": g.d.min(), "d_max": g.d.max(), "p_raw_max": g.p_raw.max()})
    out = pd.DataFrame(rows)
    out.to_csv("results/reference_draws_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
