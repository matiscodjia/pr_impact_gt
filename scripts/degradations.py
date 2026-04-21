"""Dégradations stochastiques de labels pour l'étude de robustesse.

Ce module fournit des fonctions de dégradation qui simulent des
annotations imparfaites (GT-) à partir d'annotations propres (GT*).
Deux types de dégradations sont implémentés :

1. **Morphologique** : érosion ou dilatation aléatoire du masque,
   simulant une sur- ou sous-segmentation par un annotateur.
2. **Omission** : suppression aléatoire de petites composantes
   connexes, simulant des structures oubliées par un annotateur.

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
    generate_binary_structure,
    label as label_cc,
)


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
    min_size: int = 150,
) -> np.ndarray:
    """Supprime aléatoirement des petites composantes connexes du masque.

    Simule un annotateur qui oublie de segmenter des petits vaisseaux
    ou des structures secondaires.

    Parameters
    ----------
    segmentation : np.ndarray
        Masque de segmentation de shape ``(C, *spatial_dims)``.
    prob : float
        Probabilité d'appliquer la dégradation.
    min_size : int
        Taille maximale (en voxels) des composantes éligibles à la
        suppression. Les composantes plus grandes sont préservées.

    Returns
    -------
    np.ndarray
        Masque dégradé avec certaines petites composantes supprimées.
    """
    if np.random.rand() > prob:
        return segmentation

    result = segmentation.copy()

    for c in range(result.shape[0]):
        mask = result[c] > 0.5

        if not mask.any():
            continue

        labeled_array, num_features = label_cc(mask)

        if num_features <= 1:
            continue

        comp_sizes = np.bincount(labeled_array.ravel())
        # Indices des petites composantes (exclure le fond = index 0)
        small_comps = np.where(
            (comp_sizes[1:] > 0) & (comp_sizes[1:] < min_size)
        )[0] + 1

        if len(small_comps) == 0:
            continue

        # Supprimer une composante aléatoire
        to_remove = np.random.choice(small_comps)
        mask[labeled_array == to_remove] = False

        result[c] = mask.astype(result.dtype)

    return result


def apply_combined_degradation(
    segmentation: np.ndarray,
    morpho_prob: float = 0.3,
    morpho_max_radius: int = 3,
    omission_prob: float = 0.2,
    omission_min_size: int = 150,
) -> np.ndarray:
    """Applique séquentiellement les dégradations morpho puis omission.

    Parameters
    ----------
    segmentation : np.ndarray
        Masque de segmentation de shape ``(C, *spatial_dims)``.
    morpho_prob : float
        Probabilité de la dégradation morphologique.
    morpho_max_radius : int
        Rayon maximum pour la morphologie.
    omission_prob : float
        Probabilité de l'omission.
    omission_min_size : int
        Taille seuil pour l'omission.

    Returns
    -------
    np.ndarray
        Masque après les deux dégradations.
    """
    result = apply_morpho_degradation(
        segmentation, prob=morpho_prob, max_radius=morpho_max_radius
    )
    result = apply_omission_degradation(
        result, prob=omission_prob, min_size=omission_min_size
    )
    return result
