#!/usr/bin/env python3
"""Agrège les tirages multiples de GT- (variance de la RÉALISATION du bruit).

Entrées : results/dose_response.csv (tirage de l'étude, graine 42, ligne p=0.3) et
          results/dose_response_draw*.csv (un fichier par graine, cf. dose_response.py --tag).
Sortie  : results/reference_draws_summary.csv + tableau imprimé.

Pour chaque (métrique, paire) : interaction moyenne par tirage, puis moyenne / écart-type /
min / max sur les tirages, et nombre de tirages où le signe de l'écart M0-Mk change
entre GT* et GT-. Les tirages portent sur les MÊMES 85 cas : ils ne sont pas indépendants
entre eux ; on rapporte la dispersion des moyennes, pas un intervalle de confiance.
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
                     "d_min": g.d.min(), "d_max": g.d.max()})
    out = pd.DataFrame(rows)
    out.to_csv("results/reference_draws_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
