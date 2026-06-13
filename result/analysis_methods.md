# Méthodes d'analyse — métriques, dégradations formelles et tests statistiques

## 1. Métriques d'évaluation

Chaque prédiction est comparée à une vérité terrain (GT*) ou à une GT dégradée (GT⁻) sur le même cas. Quatre métriques sont calculées par cas, indépendamment.

### CLDice (↑) — fidélité topologique

Score de Dice évalué sur les **squelettes** des deux volumes plutôt que sur les volumes pleins. On extrait le squelette de la prédiction et celui de la GT, puis on calcule deux composantes :

- **Tprec** : fraction du squelette prédit qui tombe dans le volume GT.
- **Tsens** : fraction du squelette GT qui tombe dans le volume prédit.

CLDice = moyenne harmonique de Tprec et Tsens. Sensible à la connectivité des structures tubulaires (vaisseaux) — un trou dans une artère pénalise fortement, contrairement au Dice volumétrique classique.

### HD95 (↓, mm) — distance de Hausdorff au 95ᵉ percentile

Pour chaque voxel de surface d'un volume, on calcule sa distance euclidienne au plus proche voxel de surface de l'autre volume (et inversement). HD95 = 95ᵉ percentile de l'union de ces distances. Mesure le **pire écart géométrique** tout en étant robuste aux 5 % de points les plus aberrants. Spacing voxel pris en compte → unité physique (mm).

### NSD (↑) — Normalized Surface Dice (tolérance 2 mm)

Fraction des points de surface (prédiction et GT) dont le plus proche correspondant sur l'autre surface est à ≤ 2 mm. Plus tolérant que le Dice volumétrique. Mesure d'**accord de surface** à une tolérance clinique fixée.

### Betti₀ (↓) — erreur topologique de connectivité

|N(prédiction) − N(GT)| où N est le nombre de composantes connexes 26-voisinage. Compte les **fragmentations** et les **fusions**. Complémentaire au CLDice.

---

## 2. Dégradations formelles d'annotation GT* → GT⁻

Les dégradations sont paramétrées par θ = (family, r, p, seed) et implémentées dans
`scripts/degradations.py`. Elles sont directement portées depuis l'étude
`metric_reliability_study` (voir `metric_reliability_study/docs/METHODS.md` pour la
définition formelle des estimands).

### Familles disponibles

| Famille | Mécanisme | r | p |
|---|---|---|---|
| `distal_omission` | Ouverture morpho (boule r) → retire les structures plus fines que r. Chaque structure fine (composante connexe de voxels retirés) est retirée avec Bernoulli(p) **par unité structurelle** (radius-gated). | rayon de la boule (voxels) | probabilité de retrait par composante |
| `boundary_jitter` | Déplacement de surface ±r voxels (en mm via spacing), champ lissé spatialement corrélé (level-set sur distance signée). | amplitude (mm via spacing) | fraction de surface affectée |
| `distal_truncation` | Élagage des tronçons terminaux du squelette, regonflé en tube binaire (partition par EDT). Bernoulli(p) par tronçon terminal. | longueur coupée (itérations) | probabilité par tronçon |
| `homogeneous_morpho` | **Contrôle** : érosion globale uniforme (boule r), **sans** gating radius — irréaliste. Sert à distinguer l'effet du volume retiré de l'effet de la structure du bruit. | épaisseur érodée | fraction de surface affectée |

### Calibration (points d'opération MESURÉS sur PARSE)

Les paramètres (r, p) sont choisis pour que clDice(GT⁻, GT*) ∈ [0.85, 0.90], cohérent
avec la fenêtre inter-observateur de la littérature. Mesures sur PARSE CT (n=5-8 cas,
artères pulmonaires) :

| Configuration | clDice(GT⁻, GT*) | Statut |
|---|---|---|
| **PIPELINE combiné** `distal_omission r1 p0.3` + `boundary_jitter r2 p1.0` | **0.866** | ✅ point GT⁻ retenu |
| `distal_omission` r2 p0.3 (isolé) | 0.864 | ✅ en fenêtre |
| `boundary_jitter` r3 p1.0 (isolé) | 0.889 | ✅ en fenêtre |
| `homogeneous_morpho` r1 p0.5 (contrôle) | 0.877 | ✅ en fenêtre |

**Deux faits empiriques importants :**

1. **La fenêtre clDice est pilotée par `distal_omission`.** `boundary_jitter` est quasi-invisible
   au clDice (r2 p0.5 isolé → 0.955) car clDice est une métrique de **squelette** : déplacer
   le bord d'un tube ne déplace pas sa ligne centrale. Pour calibrer le désaccord *surfacique*
   du jitter, il faut viser **NSD/HD95** en complément.
2. **Les effets se cumulent** : en pipeline, chaque famille doit être plus douce qu'isolée
   (d'où `distal_omission` r1 p0.3 dans le combiné, contre r2 p0.3 isolé).

La calibration se relance via :

```bash
python scripts/calibrate.py --noise-only
```

Elle balaye la grille (r_values, p_values) définie dans `calibration.noise_calibration`
du config et identifie les points d'opération dans la fenêtre cible.

---

## 3. Tests statistiques

### Test utilisé : Wilcoxon signed-rank apparié

Pour chaque comparaison, on dispose de N=15 cas de test évalués dans deux conditions. Les paires sont **appariées par cas**. Le test de Wilcoxon teste H₀ : la médiane des différences par paire = 0, sans hypothèse de normalité.

**Taille d'effet** : Cohen's d apparié = mean(diff) / std(diff). Convention : |d| < 0.2 négligeable, 0.2–0.5 petit, 0.5–0.8 moyen, > 0.8 grand.

**Pas de correction multiple.** Justification : 2 comparaisons pré-spécifiées par famille (par métrique × par type de question), définies a priori par le design expérimental — pas d'exploration post-hoc.

### Deux familles de tests, trois questions de recherche

#### Q1 — Pénalisation artificielle par GT⁻ d'évaluation (Famille B)

> **Question** : pour un modèle donné, le changement de scénario d'évaluation (GT* → GT⁻) provoque-t-il une chute significative de performance ?

Comparaison GT* vs GT⁻ pour chaque modèle. Si un modèle entraîné sur GT propres voit ses scores chuter significativement quand **seule la GT d'évaluation change**, alors la pénalité est imputable à la GT et non au modèle. Les dégradations d'évaluation utilisent `distal_omission` (p=1.0) + `boundary_jitter` (p=0.7), r=2 — calibré inter-observateur.

#### Q2 — Robustesse induite par l'entraînement sur GT⁻ (Famille A)

> **Question** : un modèle entraîné sur GT⁻ (Model_Minus_Stoch, `distal_omission` + `boundary_jitter` stochastiques on-the-fly) est-il plus robuste à l'évaluation sur GT⁻ qu'un modèle entraîné sur GT* (Model_Star) ?

Comparaison Model_Minus_Stoch vs Model_Star **sur GT⁻**. Le contrôle `Model_Minus_HomoMorpho` (entraîné avec `homogeneous_morpho`, dégradation irréaliste) permet de tester si la robustesse est spécifique à la **structure** du bruit d'entraînement ou liée au volume retiré.

#### Q3 — Apprentissage du biais de segmentation (pénalité asymétrique)

> **Question** : si la même distribution de bruit est appliquée à l'entraînement ET à l'évaluation, un modèle colle-t-il mieux à la GT dégradée, résultant en une pénalité d'évaluation plus faible ?

Pénalité = CLDice(GT*) − CLDice(GT⁻) par cas et par modèle. L'hypothèse : Model_Minus_Stoch (entraîné sur la même distribution que celle utilisée pour GT⁻ de test) a une pénalité plus faible que Model_Star. Model_Minus_Fixed (entraîné sur une réalisation fixe de GT⁻) est le cas intermédiaire. Testé via `noise_learning_test()` : Wilcoxon apparié sur les pénalités inter-modèles.

---

## 4. Résumé des résultats (cf. `result/statistical_tests.csv`)

### Famille A — différence entre modèles (n=15)

| Métrique | Scénario | Model_Minus | Model_Star | Δ | d | p brut | Sig. |
|---|---|---|---|---|---|---|---|
| CLDice ↑ | GT* | 0.879 | 0.877 | −0.002 | −0.09 | 0.720 | Non |
| CLDice ↑ | GT⁻ | 0.762 | 0.748 | −0.014 | −0.71 | **0.018** | **Oui** |
| HD95 ↓ | GT* | 1.900 | 1.471 | −0.429 | −0.33 | 0.110 | Non |
| HD95 ↓ | GT⁻ | 4.468 | 3.999 | −0.469 | −0.22 | 0.311 | Non |

Différence inter-modèles détectée uniquement sur CLDice en GT⁻ — Model_Minus légèrement supérieur (effet moyen, d ≈ −0.71). HD95 ne sépare pas les modèles, suggérant que la différence est topologique (connectivité) plus que géométrique.

### Famille B — pénalité d'évaluation GT* → GT⁻ (n=15)

| Métrique | Modèle | GT* | GT⁻ | Δ | d | p brut | Sig. |
|---|---|---|---|---|---|---|---|
| CLDice ↑ | Model_Minus | 0.879 | 0.762 | −0.117 | 0.62 | 0.083 | Non (limite) |
| CLDice ↑ | Model_Star | 0.877 | 0.748 | −0.129 | 0.67 | **0.041** | **Oui** |
| HD95 ↓ | Model_Minus | 1.900 | 4.468 | +2.568 | 0.67 | **0.0004** | **Oui** |
| HD95 ↓ | Model_Star | 1.471 | 3.999 | +2.528 | 0.55 | **0.0012** | **Oui** |

**Validation forte de l'hypothèse de pénalisation artificielle (Q1)** : sur HD95, les deux modèles subissent une chute hautement significative (p < 0.002) entre GT* et GT⁻, alors que **seule la GT d'évaluation a changé**. La pénalité est imputable à la qualité de la GT, pas au modèle.

*Note : ces résultats ont été obtenus avec les anciennes dégradations (morpho + omission). Ils seront mis à jour après le re-run avec les dégradations formelles (`distal_omission` + `boundary_jitter`, calibrées inter-observateur).*
