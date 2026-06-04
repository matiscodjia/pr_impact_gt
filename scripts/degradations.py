"""Dégradations stochastiques de labels pour l'étude de robustesse.

Ce module fournit des fonctions de dégradation qui simulent des
annotations imparfaites (GT-) à partir d'annotations propres (GT*).
Deux types de dégradations sont implémentés :

1. **Morphologique** : érosion ou dilatation aléatoire du masque,
   simulant une sur- ou sous-segmentation par un annotateur.
2. **Omission** : suppression des vaisseaux fins par ouverture
   morphologique, simulant les petites structures distales oubliées
   par un annotateur.

Les fonctions opèrent sur des arrays NumPy pour être compatibles
avec le pipeline de données nnU-Net (``batchgeneratorsv2``).

Examples
--------
>>> import numpy as np
>>> from degradations import apply_morpho_degradation
>>> mask = np.zeros((1, 64, 64, 64), dtype=np.float32)
>>> mask[0, 20:40, 20:40, 20:40] = 1.0
>>> degraded = apply_morpho_degradation(mask, prob=1.0, max_radius=2)
>>> degraded.shape
(1, 64, 64, 64)
"""

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_opening,
    generate_binary_structure,
)


def _ball(radius: int) -> np.ndarray:
    """Élément structurant : boule euclidienne 3D de rayon ``radius`` (voxels)."""
    r = int(radius)
    zz, yy, xx = np.ogrid[-r:r + 1, -r:r + 1, -r:r + 1]
    return (zz * zz + yy * yy + xx * xx) <= r * r


def apply_morpho_degradation(
    segmentation: np.ndarray,
    prob: float = 0.3,
    max_radius: int = 3,
) -> np.ndarray:
    """Applique une dégradation morphologique aléatoire au masque.

    Avec une probabilité ``prob``, le masque subit soit une érosion
    soit une dilatation avec un rayon aléatoire entre 1 et ``max_radius``.

    Parameters
    ----------
    segmentation : np.ndarray
        Masque de segmentation de shape ``(C, *spatial_dims)`` où C est
        le nombre de canaux (typiquement 1 pour segmentation binaire).
        Les valeurs doivent être 0 ou 1.
    prob : float
        Probabilité d'appliquer la dégradation (entre 0 et 1).
    max_radius : int
        Rayon maximum pour l'opération morphologique (en voxels).

    Returns
    -------
    np.ndarray
        Masque dégradé de même shape et dtype que l'entrée.
    """
    if np.random.rand() > prob:
        return segmentation

    result = segmentation.copy()

    for c in range(result.shape[0]):
        mask = result[c] > 0.5

        # Pas de dégradation si le masque est vide
        if not mask.any():
            continue

        radius = np.random.randint(1, max_radius + 1)
        mode = np.random.choice(["erode", "dilate"])

        struct = generate_binary_structure(mask.ndim, 1)

        if mode == "erode":
            degraded = binary_erosion(mask, structure=struct, iterations=radius)
        else:
            degraded = binary_dilation(mask, structure=struct, iterations=radius)

        result[c] = degraded.astype(result.dtype)

    return result


def apply_omission_degradation(
    segmentation: np.ndarray,
    prob: float = 0.2,
    radius: int = 1,
) -> np.ndarray:
    """Supprime les vaisseaux fins du masque par ouverture morphologique.

    Simule un annotateur qui oublie de segmenter les petits vaisseaux
    distaux. Un arbre vasculaire étant une *seule* composante connexe, on
    ne peut pas l'« oublier par morceaux » via les composantes connexes ;
    on cible donc le **calibre**. Une ouverture morphologique (érosion puis
    dilatation) par une boule de rayon ``radius`` fait disparaître toute
    structure de rayon inférieur à ``radius`` tout en préservant les gros
    troncs. Les voxels retirés (``mask & ~ouverture``) sont précisément les
    fines branches.

    Parameters
    ----------
    segmentation : np.ndarray
        Masque de segmentation de shape ``(C, *spatial_dims)``.
    prob : float
        Probabilité d'appliquer la dégradation.
    radius : int
        Rayon de la boule d'ouverture (en voxels). Plus il est grand, plus
        on remonte vers le tronc et plus on supprime de vaisseaux.

    Returns
    -------
    np.ndarray
        Masque dégradé, vaisseaux de rayon < ``radius`` supprimés.
    """
    if radius < 1 or np.random.rand() > prob:
        return segmentation

    result = segmentation.copy()
    struct = _ball(radius)

    for c in range(result.shape[0]):
        mask = result[c] > 0.5

        if not mask.any():
            continue

        # Ouverture = érosion puis dilatation par une boule de rayon `radius`.
        # L'ouverture est anti-extensive (opened ⊆ mask), donc `result[c] = opened`
        # revient à retirer exactement les voxels des vaisseaux fins.
        opened = binary_opening(mask, structure=struct)
        result[c] = opened.astype(result.dtype)

    return result


_REGISTRY = {
    "morpho": lambda seg, cfg: apply_morpho_degradation(
        seg, prob=cfg["prob"], max_radius=cfg["max_radius"]
    ),
    "omission": lambda seg, cfg: apply_omission_degradation(
        seg, prob=cfg["prob"], radius=cfg["radius"]
    ),
}


def apply_degradation_pipeline(
    segmentation: np.ndarray,
    degradation_configs: list,
) -> np.ndarray:
    """Applique séquentiellement une liste de dégradations.

    Parameters
    ----------
    segmentation : np.ndarray
        Masque de segmentation de shape ``(C, *spatial_dims)``.
    degradation_configs : list[dict]
        Liste ordonnée de configs, chaque dict ayant au minimum une clé
        ``"type"`` (ex: ``"morpho"``, ``"omission"``) et les paramètres
        spécifiques au type. Exemple ::

            [
                {"type": "morpho",   "prob": 0.3, "max_radius": 3},
                {"type": "omission", "prob": 0.2, "radius": 1},
            ]

    Returns
    -------
    np.ndarray
        Masque après application de toutes les dégradations.

    Raises
    ------
    ValueError
        Si un type de dégradation est inconnu.
    """
    result = segmentation
    for cfg in degradation_configs:
        dtype = cfg["type"]
        if dtype not in _REGISTRY:
            raise ValueError(
                f"Type de dégradation inconnu: '{dtype}'. "
                f"Disponibles: {sorted(_REGISTRY)}"
            )
        result = _REGISTRY[dtype](result, cfg)
    return result
