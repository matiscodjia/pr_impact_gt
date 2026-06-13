# Plan d'expériences — impact de la qualité de la GT (PARSE, nnU-Net)

## Principes directeurs

1. **Une famille de bruit à la fois.** Simple, débogable, attribution causale propre.
   Pas de bruits composés tant que chaque famille n'est pas comprise isolément
   (cf. `later.txt` pour la pré-sélection future des combinaisons).
2. **L'instrument doit matcher la grandeur — en type, en échelle ET en direction.**
   - bruit **topologique** (omission) → **clDice, Betti₀**
   - **magnitude** d'un bruit de bord (drift, tout μ) → **NSD@0.5 mm**
     (NSD@2 mm trop grossier — aveugle à une dérive ~1 mm ; HD95 sature sous le voxel).
   - **direction** d'un biais (drift μ≠0 = sous/sur-segmentation) → **ΔV signé**
     (`compute_volume_delta`). NSD et HD95 sont **non signées** → μ=−0.5 et μ=+0.5
     donnent le même NSD. Un biais directionnel exige un instrument directionnel.
   - **Pas de dissociation propre sur les vaisseaux fins** (vérifié, cf. Étape 0) : un
     décalage de bord ~ rayon du vaisseau le **pince** → le drift touche *réellement* la
     topologie aussi (Betti₀). Ce n'est pas un artefact à supprimer, c'est le phénomène
     réel (une erreur d'annotation fragmente). → pour le drift on rapporte **ΔV + NSD@0.5
     + Betti₀** ensemble ; pour l'omission, clDice/Betti₀ et NSD reste, lui, plat (témoin).
3. **Axe biaisé vs non biaisé** (Heller 2018 ; Yao 2023 / GSD-Net) :
   - bruit **biaisé** (omission, drift μ≠0) → pénalise (Q1) **et** s'apprend (Q3) ;
   - bruit **non biaisé** (drift μ=0) → quasi invisible, agit en augmentation (Q2).

---

## Étape 0 — Calibration IAA (avant tout entraînement)

`python scripts/calibrate_noise.py` mesure l'accord GT⁻ vs GT* (sans entraînement) sur 4
masques PARSE. Sortie : `results/noise_calibration.csv`. Résultats mesurés :

| Bruit | Paramètre IAA retenu | Accord mesuré (PARSE) | Statut |
|---|---|---|---|
| `distal_omission` | **r=2, p=0.3** | clDice **0.878** ∈ [0.85, 0.90] | ✅ retenu |
| `boundary_drift` (r1 p0.5) | **r=1, p=0.5**, μ ∈ [−0.5, +0.5] | NSD@0.5 **0.88** ∈ [0.85, 0.90] | ✅ retenu |
| `distal_truncation` | — | (non pertinent — voir ci-dessous) | ❌ **écarté (redondant)** |

**Constats de calibration (vérifiés) :**
- `distal_truncation` est **écarté car redondant avec `distal_omission`** : même axe (réduction
  biaisée du foreground distal). ⚠️ Son clDice élevé (0.97–0.99) **ne prouve PAS** qu'il est
  inoffensif — clDice ne mesure que la continuité squelettique, **aveugle à un limage de bord**.
  Mauvais instrument ; la bonne mesure aurait été une métrique de surface. On l'écarte par
  non-redondance, pas par « no-op ».
- **Vaisseaux fins = surface et topologie inséparables (phénomène réel, pas artefact).**
  Betti₀ est bien en 26-conn (N(GT*)=1 ; en 6-conn on aurait 99 = piège évité). Le drift
  fragmente *réellement* l'arbre : N(deg) ≈ 59 à r1p0.5 (Betti₀ 58), ≈152 à r2p1.0. Une
  erreur de bord ~ rayon du vaisseau le pince → c'est une vraie erreur d'annotation, on la
  garde. On opère à **r1, p0.5** (le moins agressif qui reste mesurable).
- **HD95 sature** (≈ 0–0.9 mm, < 1 voxel) → inutile pour graduer le drift ; on garde NSD@0.5.
- **NSD/HD95 sont non signées** → aveugles à la direction de μ. Pour Q3, métrique **signée
  ΔV** (`compute_volume_delta` : sous-seg < 0 < sur-seg ; monotone en μ).

---

## Modèles à entraîner

Baseline partagée + un modèle par condition (truncation économisé pour l'instant).

| Modèle | Entraîné sur | Régime | Sert |
|---|---|---|---|
| `M0_Star` | GT* propre | standard | baseline (Q1, Q2, Q3) |
| `M1_Omission` | distal_omission r2 p0.3 | on-the-fly | Q1, Q2 (axe topologique) |
| `M2_Drift_μ0` | boundary_drift r1 p0.5 μ=0 | **on-the-fly** | Q2 (robustesse) |
| `M3_Drift_μ−` | boundary_drift r1 p0.5 μ=−0.5 (figé) | **figé** | Q3 (biais sous-seg appris) |
| `M4_Drift_μ+` | boundary_drift r1 p0.5 μ=+0.5 (figé) | **figé** | Q3 (biais sur-seg appris) |

**Pourquoi le régime varie** (justifié, pas arbitraire) :
- **on-the-fly** (bruit re-tiré/epoch) pour Q2 : l'augmentation EST le ré-échantillonnage.
  Un μ=0 figé apprendrait l'ondulation gelée (pire), pas la robustesse.
- **figé** (1 réalisation/cas) pour Q3 : une cohorte d'annotateurs = un jeu cohérent ;
  cible stable → biais fittable, sans confondre avec l'augmentation. C'est le scénario réel.

---

## Q1 — La GT⁻ pénalise-t-elle artificiellement un modèle ?

**Montage.** `M0_Star` (jamais dégradé) évalué sur **GT\* puis GT⁻** — seule la référence
d'évaluation change. Wilcoxon apparié, n=15.

| GT⁻ d'éval | Métrique appariée | Attendu |
|---|---|---|
| omission (r2 p0.3) | clDice ↓, Betti₀ ↑ ; NSD reste plat (témoin) | **chute significative** |
| drift μ≠0 (r1 p0.5) | NSD@0.5 ↓, Betti₀ ↑, ΔV ≠ 0 (les trois couplés) | **chute significative** |

**Résultat attendu.** Le même modèle perd des points dès qu'on change la GT d'éval
→ pénalité imputable à la **qualité de la GT, pas au modèle**. Pour l'omission, NSD reste
plat = bon instrument prouvé ; pour le drift, surface ET topologie bougent (couplage réel
sur le fin, pas un défaut).
**Nul** = la pseudo-GT* ne pénalise pas (improbable côté topologique).

## Q2 — Le bruit non biaisé agit-il comme augmentation (robustesse) ?

**Montage.** `M2_Drift_μ0` (on-the-fly) vs `M0_Star`, **tous deux évalués sur GT\* propre**,
NSD@0.5 + Betti₀ (le drift μ=0 fragmente aussi → on suit les deux). μ=0 isole la robustesse
(aucun biais directionnel à matcher, ΔV≈0 → exclut Q3).

**Résultat attendu.** `M2 ≥ M0` sur du propre : au minimum **sans dégât** (le bruit
moyenne nulle se compense → vrai bord appris), au mieux léger bénéfice de régularisation.
**Bonus** (1 modèle de plus) : `figé μ=0` < `on-the-fly μ=0` → démontre que c'est le
ré-échantillonnage qui fait la robustesse.
**⚠️ Issue probable : Δ minuscule, sous le seuil de détection** → on ne pourra pas trancher
« pas d'effet » vs « effet sous la sensibilité métrique ». **→ motivation métrologie**
(MDD, sensibilité des métriques ; boucle avec `metric_reliability_study`).

## Q3 — Le biais de segmentation peut-il être appris ?

**Montage.** `M3_Drift_μ−` entraîné sur GT⁻ **figé** (μ<0, biais cohérent par cas).
Évalué contre deux références. Métrique **signée ΔV** (directionnelle) + NSD@0.5.
Wilcoxon apparié sur les pénalités.

| Évalué contre | Attendu pour M3 | vs M0_Star |
|---|---|---|
| **GT⁻ biaisé** (la « cohorte ») | pénalité **faible** (biais appris) ; ΔV(M3) ≈ ΔV(GT⁻) | M0 pénalisé **plus fort** |
| **GT\* vrai** | **pire** que M0 ; ΔV(M3) < 0 (sous-segmente comme appris) | — |

**Résultat attendu.** Le modèle entraîné sur le biais **colle mieux aux mauvaises données**
(pénalité réduite sur GT⁻ biaisée) **et** s'éloigne du vrai (pire sur GT\*) → biais
intériorisé. Miroir pour `M4_Drift_μ+`.
**Nul** = pénalité identique M3 vs M0 (improbable : μ est un signal constant apprenable).

---

## Statistiques (commun à toutes les questions)

- **Test** : Wilcoxon signed-rank apparié (n=15 cas test), pas d'hypothèse de normalité.
- **Taille d'effet** : Cohen's d apparié.
- **Si non significatif** : rapporter le **MDD** (plus petite différence détectable au vu
  de la variance inter-cas) — ne PAS conclure « pas d'effet ». C'est le pont métrologie.
- **Pré-enregistrement** : métrique appariée = endpoint primaire par famille. Pour
  l'omission, NSD = témoin (dissociation). Pour le drift, surface+topologie sont couplées
  (pas de témoin plat) → endpoint primaire = ΔV signé (direction).

## Récit attendu d'ensemble

Deux niveaux de « réaction » à distinguer (cf. métrologie) :
(i) **bruit ↔ métrique** (GT⁻ vs GT*) · (ii) **modèle ↔ modèle** (entraînés, sur test).

| Bruit | (i) déforme la GT ? | (ii) sépare les modèles ? | Conclusion |
|---|---|---|---|
| omission, drift μ≠0 (**biaisé**) | oui (métrique appariée) | oui | pénalise (Q1) **et** s'apprend (Q3) |
| drift μ=0 (**non biaisé**) | oui (NSD, Betti₀ à r1p0.5) | **probablement non / sous le seuil** | bénin → augmentation (Q2) → **motive la métrologie** |

**Ligne directrice : les bruits biaisés déclassent et se font apprendre.** Le bruit non
biaisé déforme bien une réalisation de GT, mais son effet sur le *modèle* (on-the-fly,
moyenné) sera petit — possiblement sous la sensibilité des métriques. Ce n'est pas un
échec : c'est ce qui justifie d'étudier la **sensibilité des métriques elles-mêmes** (MDD),
et boucle avec `metric_reliability_study`.
