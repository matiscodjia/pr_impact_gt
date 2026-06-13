# nnU-Net PARSE — Étude de robustesse aux dégradations de vérité terrain

Pipeline de segmentation des artères pulmonaires (dataset **PARSE**, CT) avec
**nnU-Net v2**, conçu pour mesurer l'impact des **dégradations d'annotation** sur
l'apprentissage. Pensé pour tourner sur **un seul GPU partagé** sujet aux
coupures : l'exécution **reprend sans perte** après une interruption et produit
des **résultats dès le premier fold**, qui s'affinent à mesure que le calcul
avance.

> **Ce README est fait pour être suivi pas à pas, sans connaissance en
> informatique.** Copiez-collez les blocs dans l'ordre. À chaque étape, ce que
> vous devez voir est indiqué.

---

## 🟢 Le plus simple (3 commandes)

Sur le cluster, dans un terminal SSH :

```bash
# 1. Installation (UNE SEULE FOIS)
bash setup_env.sh --cluster

# 2. Activer l'environnement (À CHAQUE NOUVELLE SESSION SSH)
source .env_nnunet

# 3. Tout lancer (préparation des données + entraînement + résultats)
bash run_pipeline.sh /chemin/vers/PARSE
```

C'est tout. Le pipeline prépare les données, entraîne les 3 modèles sur 5 folds,
et écrit les résultats dans `results/`. **Si ça s'interrompt** (coupure, GPU pris
par un collègue, redémarrage), relancez simplement la **même commande** : ça
repart d'où ça s'était arrêté.

Pour suivre l'avancement à tout moment, ouvrez le **tableau de bord** :

```bash
cat results/STATUS.md
```

> 💡 **Recommandé** avant le grand run : faire d'abord une **calibration** pour
> choisir le bon nombre d'époques (section [Calibration](#-calibration-recommandé)).

---

## Table des matières

1. [Prérequis](#1-prérequis)
2. [Installation (une fois)](#2-installation-une-fois)
3. [Chaque session](#3-chaque-session)
4. [Lancer le pipeline complet](#4-lancer-le-pipeline-complet)
5. [Suivre l'avancement](#5-suivre-lavancement)
6. [Reprendre après une interruption](#6-reprendre-après-une-interruption)
7. [Calibration (recommandé)](#-calibration-recommandé)
8. [Lire les résultats](#7-lire-les-résultats)
9. [Référence des commandes (toutes les options)](#8-référence-des-commandes-toutes-les-options)
10. [Configuration](#9-configuration-le-fichier-à-éditer)
11. [Dépannage](#10-dépannage)

---

## 1. Prérequis

- Un accès **SSH** au cluster avec un **GPU** (≈ 40 Go de VRAM recommandés).
- Le dataset **PARSE** quelque part sur le cluster (images CT + annotations).
- `git` disponible (pour récupérer nnU-Net). `uv` est installé automatiquement
  par le script d'installation s'il manque.

Vous n'avez **rien d'autre** à installer à la main.

---

## 2. Installation (une fois)

Depuis le dossier du projet, sur le cluster :

```bash
bash setup_env.sh --cluster
```

Ce script, automatiquement :
- crée un environnement Python isolé (`.venv`) ;
- installe PyTorch (CUDA), nnU-Net v2, et les dépendances du projet ;
- crée le fichier `.env_nnunet` (raccourci d'activation) ;
- installe les « trainers » personnalisés dans nnU-Net.

**Ce que vous devez voir à la fin :** un bloc « Vérification » affichant
`PyTorch: …`, `nnU-Net: OK`, votre `GPU`, et `Trainer: nnUNetTrainerDegraded OK`.

> Sur un Mac (pour tester la mécanique seulement, pas pour les vrais calculs) :
> `bash setup_env.sh --local`.

---

## 3. Chaque session

À **chaque** nouvelle connexion SSH, activez l'environnement :

```bash
source .env_nnunet
```

> ⚠️ **Important** : après `source .env_nnunet`, utilisez les commandes
> `python scripts/...` (et **pas** `uv run`). Le `source` active le bon
> environnement ; `uv run` en utiliserait un autre, sans nnU-Net.

Pour les calculs longs, lancez le tout dans **tmux** (la session survit à la
déconnexion SSH) :

```bash
tmux new -s nnunet          # créer la session
# ... lancer vos commandes ...
# Ctrl+B puis D  → détacher (vous pouvez fermer le SSH)
tmux attach -t nnunet       # revenir plus tard
```

---

## 4. Lancer le pipeline complet

```bash
bash run_pipeline.sh /chemin/vers/PARSE
```

Cela exécute, dans l'ordre, **6 étapes** :

| # | Étape | Ce qu'elle fait |
|---|-------|-----------------|
| 1 | `convert` | Convertit PARSE au format nnU-Net |
| 2 | `preprocess` | Prétraitement nnU-Net (Dataset100) |
| 3 | `generate_degraded` | Crée les annotations dégradées pour l'évaluation |
| 4 | `create_fixed_dataset` | Crée les datasets figés drift μ± (Dataset103/104, cf. §8 bis) |
| 5 | `preprocess_fixed` | Prétraitement des datasets figés |
| 6 | `orchestrate` | **Entraîne les modèles M0–M4 × folds**, collecte les métriques, écrit le rapport |

L'étape 6 affiche une **barre de progression** par entraînement (époques + temps
restant estimé) et un compteur d'étape `[i/N]`.

### Options de `run_pipeline.sh`

```bash
bash run_pipeline.sh /chemin/vers/PARSE [OPTIONS]
```

| Option | Effet |
|--------|-------|
| `--debug` | Run minuscule (2 époques) pour vérifier que tout s'enchaîne, sans attendre |
| `--from ÉTAPE` | Reprendre à une étape précise (ex. `--from orchestrate` saute la préparation) |
| `--tiers A` | (défaut) seulement les 3 modèles principaux. `--tiers ABC` ajoute grid + spécificité |
| `--config FICHIER` | Utiliser un autre fichier de configuration |

**Exemple — vérifier d'abord que tout marche** (quelques minutes) :

```bash
bash run_pipeline.sh /chemin/vers/PARSE --debug
```

---

## 5. Suivre l'avancement

À tout moment, dans un autre terminal (ou après `tmux attach`) :

```bash
cat results/STATUS.md          # tableau de bord lisible
```

Vous y trouverez :
- la **progression des folds** par modèle (barres `███░░ 3/5`) ;
- le **nombre de cas** déjà évalués (`n = 51 / 85`) ;
- l'état de l'orchestrateur (unités terminées / en cours / en attente) ;
- les **résultats courants** (effets, p-values, intervalles de confiance), avec
  la mention **PRÉLIMINAIRE** tant que les 5 folds ne sont pas finis.

Les journaux détaillés de chaque entraînement (utiles en cas de problème) :

```bash
ls results/logs/               # un fichier .log par entraînement
```

---

## 6. Reprendre après une interruption

**Rien de spécial à faire.** Relancez exactement la même commande :

```bash
source .env_nnunet
bash run_pipeline.sh /chemin/vers/PARSE
```

L'orchestrateur :
- **saute** ce qui est déjà terminé ;
- **reprend** un entraînement interrompu là où il s'était arrêté (dernier point
  de sauvegarde) ;
- **réessaie** automatiquement en cas d'échec (GPU plein, etc.).

---

## 🔬 Calibration (recommandé)

nnU-Net compte les époques de façon particulière (une époque = 250 itérations
fixes). **250 époques** est le défaut du projet, mais le bon nombre dépend de la
vitesse à laquelle le modèle apprend les **fines branches vasculaires** (la
topologie converge plus tard que le Dice classique). La calibration mesure ça
pour vous, **avant** le grand run.

```bash
# 1. Préparer les données (si pas déjà fait)
source .env_nnunet
python scripts/convert_parse_to_nnunet.py --data_dir /chemin/vers/PARSE --dataset_id 100
nnUNetv2_plan_and_preprocess -d 100 --verify_dataset_integrity
python scripts/generate_degraded_dataset.py \
    --dataset_dir "$nnUNet_raw/Dataset100_PARSE" --config configs/experiment_config.yaml

# 2. Lancer la calibration (1 fold, run long avec points de mesure)
python scripts/orchestrator.py --calibrate
```

**Résultat :** un nombre d'époques **recommandé** s'affiche, et une figure
`results/figures/calibration_topology.png` montre la convergence de clDice et
Betti0. Reportez ce nombre dans la configuration (`training.num_epochs`,
section [Configuration](#9-configuration-le-fichier-à-éditer)), puis lancez le
pipeline complet.

> Pour un test rapide de la mécanique de calibration : `--calibrate --debug`.

---

## 7. Lire les résultats

Tout est dans le dossier `results/` :

| Fichier | Contenu |
|---------|---------|
| `STATUS.md` | **Tableau de bord** (à lire en premier) |
| `metrics.csv` | **Table maîtresse** : une ligne par (modèle, fold, scénario, cas, métrique) avec sa provenance |
| `outcomes_oof.csv` | Les comparaisons clés (outcomes 1, 3, 4, 6) : effet, intervalle de confiance, p-value |
| `metrics_stats_oof.csv` | Tous les tests appariés (Wilcoxon + correction FDR + taille d'effet) |
| `figures/` | Figures par outcome + courbe de convergence |
| `logs/` | Journaux bruts de chaque entraînement |
| `ledger.json` | État interne de l'orchestrateur (technique) |

Les fichiers `*_oof.*` = évaluation **out-of-fold** (analyse principale, n→85).
Les fichiers `*_test.*` = évaluation sur le **jeu de test** (confirmation, n=15).

Pour régénérer le rapport et les figures à partir des données déjà calculées
(sans rien réentraîner) :

```bash
python scripts/report.py
```

---

## 8. Référence des commandes (toutes les options)

> Toujours après `source .env_nnunet`. Les arguments entre `[ ]` sont optionnels ;
> les valeurs par défaut sont indiquées.

### `orchestrator.py` — entraînement reprenable + collecte + rapport

```bash
python scripts/orchestrator.py [OPTIONS]
```

| Option | Défaut | Effet |
|--------|--------|-------|
| `--config FICHIER` | `configs/experiment_config.yaml` | Fichier de configuration |
| `--results_dir DOSSIER` | `results` | Où écrire résultats/ledger/logs |
| `--tiers LETTRES` | `A` | Modèles à exécuter : `A` (cœur), `ABC` (tout) |
| `--folds N [N ...]` | tous (0–4) | Limiter à certains folds, ex. `--folds 0 1` |
| `--models NOM [NOM ...]` | tous du tier | Limiter à certains modèles |
| `--device cuda\|mps\|cpu` | auto | Forcer le matériel |
| `--max-attempts N` | `3` | Tentatives par entraînement avant abandon |
| `--backoff SECONDES` | `30` | Attente entre deux tentatives |
| `--debug` | — | Run minuscule (2 époques) |
| `--calibrate` | — | Mode calibration (voir section dédiée) |
| `--status` | — | Affiche l'état de chaque unité et quitte |
| `--dry-run` | — | Liste les entraînements prévus sans rien lancer |
| `--no-progress` | — | Désactive la barre (logs bruts, utile hors terminal) |

Exemples :
```bash
python scripts/orchestrator.py --status                 # où en est-on ?
python scripts/orchestrator.py --dry-run --tiers ABC    # qu'est-ce qui serait lancé ?
python scripts/orchestrator.py --folds 0                # juste le fold 0 des 3 modèles
python scripts/orchestrator.py --tiers ABC              # tout (cœur + grid + spécificité)
```

### `report.py` — (re)générer le rapport et les figures

```bash
python scripts/report.py [--results_dir results] [--kind oof|test]
```

### `collect_metrics.py` — (re)calculer les métriques manquantes

```bash
python scripts/collect_metrics.py [--config FICHIER] [--output results]
```
(Lancé automatiquement par l'orchestrateur ; à utiliser seul uniquement pour
forcer une re-collecte.)

### `calibrate.py` — évaluer les points de calibration

```bash
python scripts/calibrate.py [--config FICHIER] [--results_dir results] [--device …] [--keep-preds]
```
(Lancé automatiquement par `orchestrator.py --calibrate` ; à utiliser seul si
l'entraînement de calibration est déjà fait.)

### `plot_convergence.py` — courbe de convergence (Dice par époque)

```bash
python scripts/plot_convergence.py [--trainer nnUNetTrainerStd] [--dataset 100] \
    [--out results/figures/convergence.png]
```

### `grid_search.py` — exploration des paramètres de dégradation (phase P2)

```bash
python scripts/grid_search.py --config configs/experiment_config.yaml --phase all
# --phase 1 (distal_omission seule) | 2 (boundary_jitter seule) | 3 (combinaisons) | all
# --debug pour un run court
```

### `calibrate_noise.py` — calibration IAA des paramètres de dégradation (avant le pipeline)

Applique chaque dégradation (SANS entraînement) sur quelques masques et mesure l'accord
GT⁻ vs GT* avec l'instrument **apparié au type de bruit** : clDice/Betti₀ pour les bruits
topologiques (omission, troncature), NSD@0.5 mm/HD95 pour le bruit de surface
(`boundary_drift`). Identifie les paramètres dans la fenêtre inter-observateur
(clDice ∈ [0.85, 0.90]). Justifie la magnitude du bruit sans la choisir arbitrairement.

```bash
# Calibration du bruit (CPU, quelques minutes)
python scripts/calibrate_noise.py --n-cases 4
python scripts/calibrate_noise.py --drift-mu -1 -0.5 0 0.5 1 --drift-r 2   # balayage du biais μ
```

Sortie : `results/noise_calibration.csv` — (family, r, p, mu, clDice, NSD@0.5, HD95, Betti0).
La métrique **hors-axe** doit rester plate (témoin de dissociation). Point d'opération
topologique validé : `distal_omission` r2 p0.3 → clDice 0.864. Voir `experiments_plan.md`.

### `visualize_degradations.py` — figures 2D + 3D de l'effet des bruits

Rend l'effet de chaque famille sur un masque réel.
**🔴 Rouge = voxels retirés** (bord reculé / amputation), **🔵 bleu = voxels ajoutés**
(bord avancé), gris translucide = structure conservée. Produit deux figures par cas :
`degradation_2d_<id>.png` (coupe axiale Original | Dégradé | Diff, annotée clDice + nb voxels)
et `degradation_3d_<id>.png` (rendu surfacique 3D, une vue par famille).

```bash
# Auto : 3 cas de labelsTr, 4 familles, r=2 p=0.5
python scripts/visualize_degradations.py

# Cas précis / paramètres / sous-ensemble de familles
python scripts/visualize_degradations.py --input mon_masque.nii.gz --r 2 --p 1.0 --seed 7
python scripts/visualize_degradations.py --families distal_omission boundary_drift
python scripts/visualize_degradations.py --mu -0.5            # boundary_drift biaisé (sous-seg)
python scripts/visualize_degradations.py --no-3d             # 2D seulement (rapide)
python scripts/visualize_degradations.py --ds-factor 3       # 3D plus léger
python scripts/visualize_degradations.py --dpi 300           # haute résolution
```

**Schéma du biais de dérive μ** (`boundary_drift`) — montre sous-segmentation ↔ jitter ↔
sur-segmentation, avec l'accord inter-annotateur (clDice + NSD@0.5 mm) annoté sous chaque
vue pour prouver le réalisme :

```bash
python scripts/visualize_degradations.py --mu-sweep -1 -0.5 0 0.5 1 --r 2 --p 1.0 --dpi 300
# → drift_mu_sweep_2d_<id>.png  et  drift_mu_sweep_3d_<id>.png
```

Sortie : `result/figures/degradations/`. Le spacing du NIfTI est propagé au générateur
(familles mm-aware) et respecté dans l'aspect ratio. Note : `distal_truncation` lance
`skeletonize` sur le volume plein (~1 min/cas) — restreindre avec `--families` ou `--no-3d`
pour itérer vite.

---

## 8 bis. Exécution du plan d'expériences sur le cluster — étapes dans l'ordre

Plan détaillé (modèles, questions, résultats attendus) : **`experiments_plan.md`**.
Modèles : `M0_Star` (propre), `M1_Omission` (topologique on-the-fly), `M2_Drift_mu0`
(bord non biaisé on-the-fly), `M3/M4_Drift_mu±` (biais de bord, datasets FIGÉS 103/104).
Schedule : **500 epochs** (`training.num_epochs`).

Lancer **à tour de rôle** :

```bash
# 0) À CHAQUE session
source .env_nnunet

# 1) Données → Dataset100 (propre)
python scripts/convert_parse_to_nnunet.py --data_dir /chemin/vers/PARSE --dataset_id 100
nnUNetv2_plan_and_preprocess -d 100 --verify_dataset_integrity

# 2) (optionnel) vérifier la calibration IAA du bruit
python scripts/calibrate_noise.py --n-cases 4

# 3) GT⁻ d'ÉVALUATION (scénarios : omission r2p0.3, drift μ−0.5, drift μ+0.5)
python scripts/generate_degraded_dataset.py \
    --dataset_dir "$nnUNet_raw/Dataset100_PARSE" \
    --config configs/experiment_config.yaml --seed 42

# 4) Datasets d'ENTRAÎNEMENT figés pour Q3 (drift μ±)
python scripts/create_fixed_degraded_dataset.py --dataset_id 103 \
    --source_dir "$nnUNet_raw/Dataset100_PARSE" \
    --output_dir "$nnUNet_raw/Dataset103_PARSE_DriftMuMinus" \
    --config configs/experiment_config.yaml --seed 42
nnUNetv2_plan_and_preprocess -d 103 --verify_dataset_integrity

python scripts/create_fixed_degraded_dataset.py --dataset_id 104 \
    --source_dir "$nnUNet_raw/Dataset100_PARSE" \
    --output_dir "$nnUNet_raw/Dataset104_PARSE_DriftMuPlus" \
    --config configs/experiment_config.yaml --seed 42
nnUNetv2_plan_and_preprocess -d 104 --verify_dataset_integrity

# 5) Entraînement (500 ep, reprenable) + collecte métriques + rapport — tiers A puis B
#    L'orchestrateur enchaîne M0–M4 × folds, puis collect_metrics.py et report.py.
python scripts/orchestrator.py --config configs/experiment_config.yaml --tiers AB
#    Premier passage rapide possible sur 1 fold :  --folds 0
#    Un modèle précis :  --models M1_Omission

# 6) Résultats
python scripts/report.py --config configs/experiment_config.yaml   # regénère figures + stats
cat result/statistical_tests.csv
```

---

## 9. Configuration : le fichier à éditer

Tout se règle dans **`configs/experiment_config.yaml`**. Les réglages les plus
utiles :

```yaml
training:
  num_epochs: 500          # point de convergence (recherche antérieure) ; lu par tous les trainers
  folds: [0, 1, 2, 3, 4]   # folds de la validation croisée

experiment:
  models:                  # plan M0–M4 (cf. experiments_plan.md) ; tier A avant B
    - {name: M0_Star,          trainer: nnUNetTrainerStd,                  dataset_id: 100, tier: A}
    - {name: M1_Omission,      trainer: nnUNetTrainerDegradedOmissionOnly, dataset_id: 100, tier: A}
    - {name: M2_Drift_mu0,     trainer: nnUNetTrainerDriftMu0,             dataset_id: 100, tier: A}
    - {name: M3_Drift_muMinus, trainer: nnUNetTrainerStd,                  dataset_id: 103, tier: B}
    - {name: M4_Drift_muPlus,  trainer: nnUNetTrainerStd,                  dataset_id: 104, tier: B}

evaluation:
  scenarios:               # un bruit par scénario, métrique appariée (pas de bruit composé)
    - {name: GT_star,           degradation: null}                                   # propre
    - {name: GT_minus_omission, degradation: [{family: distal_omission, r: 2, p: 0.3}]}      # Q1 topo
    - {name: GT_minus_drift_neg, degradation: [{family: boundary_drift, r: 1, p: 0.5, mu: -0.5}]}  # Q3 μ<0
    - {name: GT_minus_drift_pos, degradation: [{family: boundary_drift, r: 1, p: 0.5, mu: 0.5}]}   # Q3 μ>0
```

> ⚠️ **Calibration mesurée, pas supposée.** Apparier l'instrument au bruit : omission →
> clDice/Betti₀ ; drift → NSD@0.5 (magnitude) + ΔV signé (direction). HD95 sature, clDice est
> aveugle au bord. Re-vérifier après tout changement de (r, p, μ) via `scripts/calibrate_noise.py`.

**Familles de dégradation disponibles** (dans `scripts/degradations.py`) :

| Famille | Usage | `r` (échelle) | `p` (sévérité) |
|---|---|---|---|
| `distal_omission` | Retire les structures fines < r (Bernoulli **par composante**) | rayon de l'ouverture (vx) | proba **par branche fine** |
| `distal_truncation` | Élague les tronçons terminaux du squelette (Bernoulli **par tronçon**) | longueur coupée (itér.) | proba **par tronçon terminal** |
| `boundary_drift` | Déplace la surface : **dérive systématique `μ` + jitter** (champ lissé, mm-aware) | amplitude (mm via spacing) | **fraction de surface** déplacée |
| `homogeneous_morpho` | **Contrôle** irréaliste : érosion uniforme sans gating | épaisseur érodée (vx) | **fraction de surface** affectée |

> `boundary_jitter` reste accepté : c'est un **alias de `boundary_drift` avec μ=0**.
> `boundary_drift` **généralise et remplace** le jitter en ajoutant le biais de dérive `μ`.

#### Le paramètre `μ` de `boundary_drift` — biais de frontière (état de l'art : Yao 2023 / GSD-Net)

Le déplacement du bord vaut `amplitude·(μ + fluctuation)`, qui **décompose** le bruit de bord
en une part systématique (μ) et une part irrégulière de moyenne nulle :

| `μ` | Effet sur le bord | Nature | Question de recherche |
|---|---|---|---|
| **μ = 0** | oscille symétriquement | non biaisé → se compense → **augmentation** | **Q2** (robustesse) |
| **μ < 0** | recule (sous-segmente) | biais systématique → **apprenable** | **Q3** (biais appris) |
| **μ > 0** | avance (sur-segmente) | biais systématique → **apprenable** | **Q3** (biais appris) |

C'est la distinction **biaisé vs non biaisé** de la littérature (Heller 2018) : un réseau est
robuste au jitter non biaisé (μ=0), mais peut **apprendre** un biais systématique (μ≠0). Un
seul bouton μ balaie tout l'axe. Visualiser : `--mu-sweep -1 -0.5 0 0.5 1` (cf. §8).

#### Sémantique de θ = (`r`, `p`, `μ`, `seed`) — à lire attentivement

- **Politique d'application : 100 %.** Le bruit est appliqué **systématiquement à toutes
  les images** (et à tous les batches d'entraînement). `p` n'est **jamais** une probabilité
  d'appliquer ou non la dégradation à une image — des garanties dans le code assurent qu'au
  moins une structure/un voxel est toujours touché dès que `p > 0`.

- **⚠️ `p` n'a pas le même sens selon la famille** — c'est la nuance clé :
  - **Bruits structurels** (`distal_omission`, `distal_truncation`) : `p` est une probabilité
    **Bernoulli par unité topologique**. `p = 0.3` ⇒ « ~30 % des **branches fines** (resp.
    tronçons) retirées », **pas** 30 % de l'aire de l'image. Les structures *candidates* sont
    fixées par `r` (l'ouverture/élagage), `p` décide *combien* parmi elles tombent.
  - **Bruits de surface** (`boundary_drift`, `homogeneous_morpho`) : `p` est bien une
    **fraction de la surface** (top-`p` des voxels de surface) — proche d'un « % de région ».

- **`r` = échelle** : ce qui compte comme « fin » (omission), l'amplitude de la dérive/jitter
  (en mm via le spacing du NIfTI), l'épaisseur érodée, la longueur élaguée.

- **`μ` = biais de dérive** (`boundary_drift` uniquement) : 0 = jitter non biaisé (augmentation,
  Q2) ; ≠0 = biais systématique apprenable (Q3). Défaut 0 → rétro-compatible avec le jitter.

- **`seed` = réalisation aléatoire + reproductibilité** : fixe *où* exactement le bruit frappe
  (quelles branches, quelles régions de surface, quel motif de jitter). Même `(family, r, p, μ, seed)`
  ⇒ sortie identique au voxel près.
  - Dataset101 fixe + GT⁻ d'évaluation : **seed fixe par cas** → GT⁻ figé et reproductible (tests appariés).
  - `nnUNetTrainerDegraded` (on-the-fly) : **seed aléatoire à chaque step** → le lieu varie d'epoch
    en epoch = diversité des réalisations = effet data augmentation.

> Vérifier l'effet visuel de chaque famille (rouge = retiré, bleu = ajouté) :
> `python scripts/visualize_degradations.py` (cf. §8).

Après avoir changé `num_epochs`, relancez simplement le pipeline : les
entraînements déjà terminés ne sont pas refaits.

---

## 10. Dépannage

| Symptôme | Cause / solution |
|----------|------------------|
| `No module named 'nnunetv2'` ou `nnUNetv2_train introuvable` | Vous avez oublié `source .env_nnunet`, **ou** vous avez utilisé `uv run` au lieu de `python`. Faites `source .env_nnunet` puis `python scripts/...`. |
| `warning: VIRTUAL_ENV ... does not match ... and will be ignored` | Le dossier du projet a été **déplacé** depuis l'installation. Relancez `bash setup_env.sh --cluster` pour recréer l'environnement au bon endroit. |
| `ERREUR: variables nnU-Net non définies` | Lancez `source .env_nnunet` avant le pipeline. |
| L'entraînement s'arrête (GPU pris, coupure) | Normal sur GPU partagé. Relancez la même commande : ça reprend tout seul. |
| Je veux voir ce qui se passe en détail | `cat results/logs/<nom>.log` (sortie brute de nnU-Net). |
| Le GPU n'est pas détecté | Forcez avec `--device cuda` sur `orchestrator.py`, et vérifiez `nvidia-smi`. |
| Je ne sais plus où ça en est | `python scripts/orchestrator.py --status` et `cat results/STATUS.md`. |

---

## Questions de recherche (rappel)

Plan détaillé (montages, résultats attendus, tests stat) : **`experiments_plan.md`**.

| # | Question | Modèles | Métrique appariée | Tier |
|---|---|---|---|---|
| Q1 | **Pénalisation artificielle** : évaluer sur GT⁻ pénalise-t-il injustement ? | `M0_Star` : GT* vs GT⁻ | clDice/Betti₀ (omission) | A |
| Q2 | **Robustesse induite** : bruit non biaisé (μ=0) = augmentation ? | `M2_Drift_mu0` vs `M0_Star` sur GT* | NSD@0.5 | A |
| Q3 | **Biais appris** : un biais de bord (μ≠0) s'apprend-il ? | `M3/M4_Drift_mu±` vs `M0_Star` | ΔV signé | B |

`--tiers AB` lance tout. `distal_truncation` écarté (redondant avec omission) ;
`homogeneous_morpho` n'est plus utilisé (contrôle de l'ancien design).
