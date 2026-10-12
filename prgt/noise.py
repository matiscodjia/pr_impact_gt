"""Noise operator N(GT*; theta): turns the clean reference GT* into a degraded reference GT-.

theta = (family, r, p, mu, seed). Four families, ported from the generator of the
metric_reliability_study project:

  family             | mechanism                                          | r                | p / mu
  ------------------ | -------------------------------------------------- | ---------------- | ------------------------------
  distal_omission    | morphological opening (ball r) + Bernoulli per     | radius (voxels)  | p = removal probability per
                     | removed component: drops structures thinner than r |                  | structure
  boundary_drift     | surface displacement = drift (mu) + smooth jitter  | amplitude (r x   | p = fraction of the surface;
                     |                                                    | mean spacing, mm)| mu = bias
  distal_truncation  | pruning of terminal skeleton branches              | length cut       | p = Bernoulli per branch
  homogeneous_morpho | unrealistic control: uniform global erosion        | eroded thickness | p = fraction of the shell

``boundary_jitter`` is an alias of ``boundary_drift`` with mu = 0 (backward compatibility).

The bias mu (boundary_drift only) sets the direction of the boundary displacement:
    mu = 0  pure zero-mean jitter: averages out, acts as augmentation
    mu < 0  systematic under-segmentation (erodes the boundary): learnable by a network
    mu > 0  systematic over-segmentation (dilates the boundary): learnable by a network

Calibration target: clDice(GT-, GT*) in [0.85, 0.90], the inter-observer window.
r = 2, p in [0.3, 0.5] is the operating point validated on PARSE.

Configuration entries (one dict per family, applied in order):
    {"family": "distal_omission",   "r": 2, "p": 0.3}
    {"family": "boundary_drift",    "r": 2, "p": 1.0, "mu": 0.0}    # jitter (augmentation)
    {"family": "boundary_drift",    "r": 2, "p": 1.0, "mu": -0.5}   # under-segmentation bias
    {"family": "homogeneous_morpho", "r": 2, "p": 0.5}              # control

Every family guarantees that a non-empty mask is changed (at least one structure or one
boundary voxel), so that 100 % of the cases are degraded whatever p.

Main API:
    apply_noise(mask, family, r, p, seed, spacing, corr_len, mu) -> uint8 mask
    apply_noise_configs(segmentation, configs, seed_base, spacing, corr_len) -> array
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

CONNECTIVITY_26 = generate_binary_structure(3, 3)


# ---------------------------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------------------------

def ball(radius: float) -> np.ndarray:
    """Boolean ball structuring element of integer radius ``round(radius)``."""
    r = int(round(radius))
    if r < 1:
        return np.ones((1, 1, 1), bool)
    zz, yy, xx = np.ogrid[-r:r + 1, -r:r + 1, -r:r + 1]
    return (zz * zz + yy * yy + xx * xx) <= r * r


def _smoothed_field(shape, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Gaussian-smoothed white noise, rescaled to unit standard deviation."""
    n = gaussian_filter(rng.standard_normal(shape).astype(np.float32), sigma=sigma)
    s = n.std()
    return n / s if s > 1e-8 else n


def _coverage_gate(shape, p: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Spatially smooth selection of a fraction p of the whole volume."""
    if p >= 1.0:
        return np.ones(shape, bool)
    if p <= 0.0:
        return np.zeros(shape, bool)
    g = _smoothed_field(shape, sigma, rng)
    return g >= np.quantile(g, 1.0 - p)


def _surface_coverage_gate(surface: np.ndarray, p: float, sigma: float,
                           rng: np.random.Generator) -> np.ndarray:
    """Spatially smooth selection of a fraction p of the SURFACE voxels.

    The quantile is taken over surface voxels only, so the selected fraction does not
    depend on how small the structure is in the volume (a volume-wide gate misses the
    surface of thin tubes, which fill < 1 % of the volume). At least one voxel is
    selected when p > 0 and the surface is not empty.
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
    if not selected.any():
        best = int(np.argmax(g[surface]))
        idx = np.where(surface.ravel())[0][best]
        flat = np.zeros(surface.size, bool)
        flat[idx] = True
        selected = flat.reshape(surface.shape)
    return selected


def _bernoulli_components(component_map: np.ndarray, n: int, p: float,
                          rng: np.random.Generator) -> np.ndarray:
    """Mask of the components drawn for removal, one Bernoulli(p) draw per component.

    If p > 0 and n > 0, at least one component is always selected.
    """
    if n == 0 or p <= 0.0:
        return np.zeros(component_map.shape, bool)
    if p >= 1.0:
        selected = np.ones(n + 1, bool)
        selected[0] = False
        return selected[component_map]
    selected = np.zeros(n + 1, bool)
    selected[1:] = rng.random(n) < p
    if not selected[1:].any():
        selected[int(rng.integers(1, n + 1))] = True
    return selected[component_map]


# ---------------------------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------------------------

def _distal_omission(mask, r, p, rng, spacing, corr_len):
    """Remove structures thinner than r (morphological opening with a ball of radius r).

    Each connected piece removed by the opening is dropped with probability p.
    """
    mask = mask > 0.5
    if r < 1 or not mask.any():
        return mask.astype(np.uint8)
    opened = binary_opening(mask, structure=ball(r))
    removed = mask & ~opened
    comp, n = label(removed, structure=CONNECTIVITY_26)
    selected = _bernoulli_components(comp, n, p, rng)
    return (mask & ~selected).astype(np.uint8)


def _boundary_drift(mask, r, p, rng, spacing, corr_len, mu=0.0):
    """Displace the surface by a systematic drift (mu) plus a smooth zero-mean jitter.

    The signed boundary displacement is

        displacement(x) = amplitude_mm * (mu + fluctuation(x))

    with amplitude_mm = r * mean spacing and fluctuation a smoothed field of mean 0 and
    standard deviation 1, applied on a fraction p of the surface. mu is a constant bias
    (mu < 0 erodes, mu > 0 dilates); the fluctuation averages out. The level set phi
    (signed distance, < 0 inside) is shifted by the displacement. If the mask comes out
    unchanged, one boundary voxel is flipped in the direction of mu.
    """
    mask = mask > 0.5
    if r < 1 or not mask.any():
        return mask.astype(np.uint8)
    phi = (distance_transform_edt(~mask, sampling=spacing)
           - distance_transform_edt(mask, sampling=spacing)).astype(np.float32)
    fluctuation = _smoothed_field(mask.shape, corr_len, rng)   # mean ~0, std 1
    surface = np.abs(phi) <= float(np.max(spacing))            # boundary voxels (+/- 1 voxel)
    gate = _surface_coverage_gate(surface, p, corr_len, rng).astype(np.float32)
    amplitude_mm = r * float(np.mean(spacing))
    displacement = amplitude_mm * (mu + fluctuation) * gate
    result = (phi - displacement <= 0).astype(np.uint8)
    if np.array_equal(result, mask.astype(np.uint8)):
        erode = mu <= 0.0
        candidates = (gate > 0) & (phi < 0 if erode else phi > 0)
        if candidates.any():
            z, y, x = np.argwhere(candidates)[rng.integers(candidates.sum())]
            result[z, y, x] = 0 if erode else 1
    return result


def _distal_truncation(mask, r, p, rng, spacing, corr_len):
    """Prune terminal skeleton branches and remove the tube around each pruned piece.

    r = number of pruning iterations from the end points; each pruned piece is removed
    with probability p.
    """
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

    comp, n = label(cut_skel, structure=CONNECTIVITY_26)
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
    """Unrealistic control: uniform erosion (ball r), not gated by vessel radius.

    Applied to a fraction p of the eroded shell (not of the whole volume).
    """
    mask = mask > 0.5
    if r < 1 or not mask.any():
        return mask.astype(np.uint8)
    eroded = binary_erosion(mask, structure=ball(r))
    shell = mask & ~eroded
    if not shell.any():
        return mask.astype(np.uint8)
    selected = shell & _surface_coverage_gate(shell, p, corr_len, rng)
    return (mask & ~selected).astype(np.uint8)


FAMILIES = {
    "distal_omission":    _distal_omission,
    "boundary_drift":     _boundary_drift,
    "boundary_jitter":    _boundary_drift,   # backward-compatible alias: jitter = drift with mu = 0
    "distal_truncation":  _distal_truncation,
    "homogeneous_morpho": _homogeneous_morpho,
}


# ---------------------------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------------------------

def apply_noise(mask: np.ndarray, family: str, r: float, p: float, seed: int,
                spacing: tuple = (1.0, 1.0, 1.0), corr_len: float = 4.0,
                mu: float = 0.0) -> np.ndarray:
    """N(GT*; theta) -> GT- as uint8. (family, r, p, mu, seed) fixes the output.

    mu is only used by boundary_drift and ignored by the other families.
    """
    if family not in FAMILIES:
        raise ValueError(
            f"Unknown noise family '{family}'. "
            f"Available: {sorted(FAMILIES)}"
        )
    rng = np.random.default_rng(seed)
    fn = FAMILIES[family]
    needs_mu = fn is _boundary_drift

    def _call(m):
        return fn(m, r, p, rng, spacing, corr_len, mu) if needs_mu \
            else fn(m, r, p, rng, spacing, corr_len)

    # The noise is local to the surface: compute it on the bounding box of the structure
    # plus a margin covering r, corr_len and the largest drift, not on the full 512^3
    # volume. The result is the same as on the full volume.
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


def apply_noise_configs(
    segmentation: np.ndarray,
    noise_configs: list,
    seed_base: int = 0,
    spacing: tuple = (1.0, 1.0, 1.0),
    corr_len: float = 4.0,
) -> np.ndarray:
    """Apply a list of noise families, in order, to every channel of a segmentation.

    Parameters
    ----------
    segmentation : np.ndarray
        Array of shape (C, *spatial_dims) with values 0/1.
    noise_configs : list[dict]
        Each dict has ``"family"``, ``"r"`` and ``"p"``; ``"mu"`` (drift bias,
        boundary_drift) and ``"seed"`` are optional (defaults: mu = 0.0,
        seed = seed_base + index of the entry). Example::

            [
                {"family": "distal_omission", "r": 2, "p": 0.3},
                {"family": "boundary_drift",  "r": 2, "p": 1.0, "mu": -0.5},
            ]

    seed_base : int
        Base seed, used when an entry has no ``"seed"``.
    spacing : tuple
        Voxel size in mm, from the NIfTI header (the families work in mm).
    corr_len : float
        Correlation length (voxels) of the smoothed fields.

    Returns
    -------
    np.ndarray
        Degraded segmentation, same shape and dtype as the input.
    """
    result = segmentation.copy()
    for i, cfg in enumerate(noise_configs):
        family = cfg.get("family")
        if family is None:
            raise ValueError(
                f"Missing 'family' key in noise config {cfg}. "
                f"Available families: {sorted(FAMILIES)}"
            )
        r = float(cfg["r"])
        p = float(cfg["p"])
        mu = float(cfg.get("mu", 0.0))
        seed = int(cfg.get("seed", seed_base + i))
        for c in range(result.shape[0]):
            mask_3d = result[c]
            result[c] = apply_noise(mask_3d, family, r, p, seed, spacing, corr_len, mu).astype(result.dtype)
    return result
