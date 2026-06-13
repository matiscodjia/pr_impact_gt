"""Dégradations formelles d'annotation GT* → GT⁻.

4 familles, paramétrisation θ = (family, r, p, seed).
Portées directement depuis le générateur de l'étude metric_reliability_study.

  family            | mécanisme                                       | r              | p / mu
  ----------------- | ----------------------------------------------- | -------------- | ------------------------------------
  distal_omission   | ouverture morpho (boule r) + Bernoulli/composante| rayon (vx)     | p = proba de retrait par structure
  boundary_drift    | dérive (mu) + jitter (champ lissé) de la surface | amplitude (mm) | p = fraction surface ; mu = biais
  distal_truncation | élagage tronçons terminaux du squelette          | longueur coupée| p = Bernoulli par tronçon
  homogeneous_morpho| contrôle irréaliste : érosion globale uniforme   | épaisseur érodée| p = fraction de surface affectée

`boundary_jitter` est un ALIAS de `boundary_drift` avec mu=0 (rétro-compatibilité).

Le paramètre mu (boundary_drift uniquement) règle le BIAIS de la dérive du bord :
    mu = 0  → jitter pur, moyenne nulle → se compense → augmentation        (Q2)
    mu < 0  → sous-segmentation systématique (érode le bord) → APPRENABLE    (Q3)
    mu > 0  → sur-segmentation systématique (dilate le bord)  → APPRENABLE   (Q3)

Calibration : viser clDice(GT⁻, GT*) ∈ [0.85, 0.90] (fenêtre inter-observateur) pour
justifier la magnitude.  r=2, p∈[0.3, 0.5] est le point d'opération validé sur PARSE.

Usage dans la config (clés) :
    {"family": "distal_omission",  "r": 2, "p": 0.3}
    {"family": "boundary_drift",   "r": 2, "p": 1.0, "mu": 0.0}    # jitter (augmentation)
    {"family": "boundary_drift",   "r": 2, "p": 1.0, "mu": -0.5}   # biais sous-seg (apprenable)
    {"family": "homogeneous_morpho","r": 2,"p": 0.5}               # contrôle

API principale :
    generate(mask, family, r, p, seed, spacing, corr_len, mu) → uint8 array
    apply_degradation_pipeline(segmentation, configs, seed_base, spacing, corr_len) → ndarray
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import (
    binary_erosion,
    binary_opening,
    convolve,
    distance_transform_edt,
    gaussian_filter,
    generate_binary_structure,
    label,
)
from skimage.morphology import skeletonize

_CONN3 = generate_binary_structure(3, 3)


# ── Primitives ─────────────────────────────────────────────────────────────────

def _ball(radius: float) -> np.ndarray:
    r = int(round(radius))
    if r < 1:
        return np.ones((1, 1, 1), bool)
    zz, yy, xx = np.ogrid[-r:r + 1, -r:r + 1, -r:r + 1]
    return (zz * zz + yy * yy + xx * xx) <= r * r


def _smoothed_field(shape, sigma: float, rng: np.random.Generator) -> np.ndarray:
    n = gaussian_filter(rng.standard_normal(shape).astype(np.float32), sigma=sigma)
    s = n.std()
    return n / s if s > 1e-8 else n


def _coverage_gate(shape, p: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    if p >= 1.0:
        return np.ones(shape, bool)
    if p <= 0.0:
        return np.zeros(shape, bool)
    g = _smoothed_field(shape, sigma, rng)
    return g >= np.quantile(g, 1.0 - p)


def _surface_coverage_gate(surface: np.ndarray, p: float, sigma: float,
                            rng: np.random.Generator) -> np.ndarray:
    """Coverage gate calculé sur les VOXELS DE SURFACE.

    Garantit que p fraction des voxels de surface est sélectionnée (au moins 1
    quand p > 0 et la surface est non vide) — indépendamment de la taille du
    masque dans le volume global.  Résout le biais du coverage_gate volumique
    qui rate les surfaces de structures fines (tube < 1 % du volume).
    """
    if p >= 1.0:
        return surface.copy()
    if p <= 0.0:
        return np.zeros(surface.shape, bool)
    if not surface.any():
        return np.zeros(surface.shape, bool)
    g = _smoothed_field(surface.shape, sigma, rng)
    surface_vals = g[surface]
    threshold = float(np.quantile(surface_vals, 1.0 - p))
    selected = surface & (g >= threshold)
    # Garantie : si p > 0 et surface non vide, au moins un voxel toujours sélectionné
    if not selected.any():
        best = int(np.argmax(g[surface]))
        idx = np.where(surface.ravel())[0][best]
        flat = np.zeros(surface.size, bool)
        flat[idx] = True
        selected = flat.reshape(surface.shape)
    return selected


def _bernoulli_components(component_map: np.ndarray, n: int, p: float,
                          rng: np.random.Generator) -> np.ndarray:
    """Sélectionne les composantes à retirer avec Bernoulli(p) par composante.

    Garantie : si p > 0 et n > 0, au moins une composante est toujours sélectionnée —
    100 % des images avec structures fines sont dégradées, quel que soit p.
    """
    if n == 0 or p <= 0.0:
        return np.zeros(component_map.shape, bool)
    if p >= 1.0:
        selected = np.ones(n + 1, bool)
        selected[0] = False
        return selected[component_map]
    selected = np.zeros(n + 1, bool)
    selected[1:] = rng.random(n) < p
    # Garantie 100 % : si aucune composante n'a été tirée, en forcer une au hasard
    if not selected[1:].any():
        selected[int(rng.integers(1, n + 1))] = True
    return selected[component_map]


# ── Familles ───────────────────────────────────────────────────────────────────

def _distal_omission(mask, r, p, rng, spacing, corr_len):
    """Ouverture morpho (boule r) → supprime les structures plus fines que r.
    Radius-gated par construction. Chaque structure fine retirée avec proba p."""
    mask = mask > 0.5
    if r < 1 or not mask.any():
        return mask.astype(np.uint8)
    opened = binary_opening(mask, structure=_ball(r))
    removed = mask & ~opened
    comp, n = label(removed, structure=_CONN3)
    selected = _bernoulli_components(comp, n, p, rng)
    return (mask & ~selected).astype(np.uint8)


def _boundary_drift(mask, r, p, rng, spacing, corr_len, mu=0.0):
    """Déplacement de la surface = dérive systématique (mu) + jitter irrégulier.

    Généralise le jitter de bord (cas mu=0). Le déplacement signé de la frontière vaut

        deplacement(x) = amplitude_mm · (mu + fluctuation(x))

    et se décompose en deux régimes complémentaires :
      • mu             → BIAIS systématique constant → APPRENABLE par un modèle (Q3).
                         mu < 0 sous-segmente (érode le bord) ; mu > 0 sur-segmente.
      • fluctuation(x) → champ lissé de MOYENNE NULLE → se compense → augmentation (Q2).

    p = fraction de surface affectée ; amplitude_mm = r · spacing moyen (mm-aware).
    On déplace le level-set phi (distance signée) : phi < 0 dedans, > 0 dehors.
    Garantie 100 % : si le masque reste inchangé, on force un voxel de bord dans le
    sens de mu (érosion si mu ≤ 0, dilatation sinon)."""
    mask = mask > 0.5
    if r < 1 or not mask.any():
        return mask.astype(np.uint8)
    phi = (distance_transform_edt(~mask, sampling=spacing)
           - distance_transform_edt(mask, sampling=spacing)).astype(np.float32)
    fluctuation = _smoothed_field(mask.shape, corr_len, rng)   # moyenne ≈ 0, écart-type 1
    surface = np.abs(phi) <= float(np.max(spacing))            # voxels du bord (± 1 voxel)
    gate = _surface_coverage_gate(surface, p, corr_len, rng).astype(np.float32)
    amplitude_mm = r * float(np.mean(spacing))
    displacement = amplitude_mm * (mu + fluctuation) * gate
    result = (phi - displacement <= 0).astype(np.uint8)
    # Garantie 100 % : aucun changement (mu=0 et fluctuation de même signe partout) → forcer.
    if np.array_equal(result, mask.astype(np.uint8)):
        erode = mu <= 0.0
        candidates = (gate > 0) & (phi < 0 if erode else phi > 0)
        if candidates.any():
            z, y, x = np.argwhere(candidates)[rng.integers(candidates.sum())]
            result[z, y, x] = 0 if erode else 1
    return result


def _distal_truncation(mask, r, p, rng, spacing, corr_len):
    """Élagage des segments terminaux du squelette, regonflé en tube binaire.
    r = longueur (itérations) coupée près des extrémités ; Bernoulli(p) par tronçon."""
    mask = mask > 0.5
    if r < 1 or not mask.any():
        return mask.astype(np.uint8)
    skel = skeletonize(mask)
    if not skel.any():
        return mask.astype(np.uint8)

    nb_struct = np.ones((3, 3, 3), np.uint8)
    nb_struct[1, 1, 1] = 0
    pruned = skel.copy()
    for _ in range(int(round(r))):
        deg = convolve(pruned.astype(np.uint8), nb_struct, mode="constant", cval=0)
        endpoints = pruned & (deg == 1)
        if not endpoints.any():
            break
        pruned &= ~endpoints
    cut_skel = skel & ~pruned
    if not cut_skel.any():
        return mask.astype(np.uint8)

    comp, n = label(cut_skel, structure=_CONN3)
    selected_cut = _bernoulli_components(comp, n, p, rng)
    if not selected_cut.any():
        return mask.astype(np.uint8)
    kept_skel = skel & ~selected_cut

    d_cut = distance_transform_edt(~selected_cut, sampling=spacing)
    d_kept = (distance_transform_edt(~kept_skel, sampling=spacing)
              if kept_skel.any() else np.full(mask.shape, np.inf))
    tube = mask & (d_cut < d_kept)
    return (mask & ~tube).astype(np.uint8)


def _homogeneous_morpho(mask, r, p, rng, spacing, corr_len):
    """Contrôle irréaliste : érosion globale uniforme (boule r), sans gating radius.
    Appliquée sur une fraction p de la COQUILLE érodée (pas du volume global)."""
    mask = mask > 0.5
    if r < 1 or not mask.any():
        return mask.astype(np.uint8)
    eroded = binary_erosion(mask, structure=_ball(r))
    shell = mask & ~eroded
    if not shell.any():
        return mask.astype(np.uint8)
    selected = shell & _surface_coverage_gate(shell, p, corr_len, rng)
    return (mask & ~selected).astype(np.uint8)


_REGISTRY = {
    "distal_omission":    _distal_omission,
    "boundary_drift":     _boundary_drift,
    "boundary_jitter":    _boundary_drift,   # alias rétro-compatible : jitter ≡ drift mu=0
    "distal_truncation":  _distal_truncation,
    "homogeneous_morpho": _homogeneous_morpho,
}


def generate(mask: np.ndarray, family: str, r: float, p: float, seed: int,
             spacing: tuple = (1.0, 1.0, 1.0), corr_len: float = 4.0,
             mu: float = 0.0) -> np.ndarray:
    """N(GT* ; θ) → GT⁻ uint8.  (family, r, p, mu, seed) fixe la sortie de façon reproductible.

    mu n'est consommé que par boundary_drift (biais de dérive du bord) ; ignoré ailleurs.
    """
    if family not in _REGISTRY:
        raise ValueError(
            f"Famille de dégradation inconnue : '{family}'. "
            f"Disponibles : {sorted(_REGISTRY)}"
        )
    rng = np.random.default_rng(seed)
    fn = _REGISTRY[family]
    needs_mu = fn is _boundary_drift

    def _call(m):
        return fn(m, r, p, rng, spacing, corr_len, mu) if needs_mu \
            else fn(m, r, p, rng, spacing, corr_len)

    # Restreindre le calcul à la bbox de la structure (+marge) : la dégradation est LOCALE
    # à la surface — inutile de calculer EDT/champs sur tout le volume (souvent 512³).
    # La marge couvre r, corr_len et l'amplitude max (mu) → résultat équivalent au plein.
    mask_bool = mask > 0.5
    if not mask_bool.any():
        return _call(mask).astype(np.uint8)
    margin = int(2 * corr_len + r * (3 + 2 * abs(mu)) + 6)
    coords = np.argwhere(mask_bool)
    lo = np.maximum(coords.min(0) - margin, 0)
    hi = np.minimum(coords.max(0) + margin + 1, mask.shape)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    out = np.zeros(mask.shape, dtype=np.uint8)
    out[sl] = _call(mask[sl]).astype(np.uint8)
    return out


def apply_degradation_pipeline(
    segmentation: np.ndarray,
    degradation_configs: list,
    seed_base: int = 0,
    spacing: tuple = (1.0, 1.0, 1.0),
    corr_len: float = 4.0,
) -> np.ndarray:
    """Applique séquentiellement une liste de dégradations formelles sur un masque.

    Parameters
    ----------
    segmentation : np.ndarray
        Masque de shape (C, *spatial_dims). Les valeurs doivent être 0/1.
    degradation_configs : list[dict]
        Chaque dict doit contenir ``"family"``, ``"r"``, ``"p"``.
        ``"mu"`` (biais de dérive, boundary_drift) et ``"seed"`` sont optionnels
        (défauts : mu=0.0 ; seed=seed_base + index de la famille).
        Exemple ::

            [
                {"family": "distal_omission", "r": 2, "p": 0.3},
                {"family": "boundary_drift",  "r": 2, "p": 1.0, "mu": -0.5},
            ]

    seed_base : int
        Graine de base. Si ``"seed"`` est absent du dict, utilise seed_base + index.
    spacing : tuple
        Espacement voxel (mm) — propagé depuis le header NIfTI pour les familles mm-aware.
    corr_len : float
        Longueur de corrélation (voxels) pour les champs lissés (boundary_jitter,
        homogeneous_morpho).

    Returns
    -------
    np.ndarray
        Masque dégradé de même shape et dtype que l'entrée.
    """
    result = segmentation.copy()
    for i, cfg in enumerate(degradation_configs):
        family = cfg.get("family")
        if family is None:
            raise ValueError(
                f"Clé 'family' manquante dans la config de dégradation : {cfg}. "
                "Remplacez 'type' par 'family' et utilisez les familles formelles : "
                f"{sorted(_REGISTRY)}"
            )
        r = float(cfg["r"])
        p = float(cfg["p"])
        mu = float(cfg.get("mu", 0.0))   # biais de dérive (boundary_drift) ; 0 par défaut
        seed = int(cfg.get("seed", seed_base + i))
        for c in range(result.shape[0]):
            mask_3d = result[c]
            result[c] = generate(mask_3d, family, r, p, seed, spacing, corr_len, mu).astype(result.dtype)
    return result
