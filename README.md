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
| 4 | `create_fixed_dataset` | Crée le dataset à annotations dégradées fixes (Dataset101) |
| 5 | `preprocess_fixed` | Prétraitement du Dataset101 |
| 6 | `orchestrate` | **Entraîne les 3 modèles × 5 folds**, collecte les métriques, écrit le rapport |

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
python scripts/plot_convergence.py [--trainer nnUNetTrainer250] [--dataset 100] \
    [--out results/figures/convergence.png]
```

### `grid_search.py` — exploration des paramètres de dégradation (phase P2)

```bash
python scripts/grid_search.py --config configs/experiment_config.yaml --phase all
# --phase 1 (morpho) | 2 (omission) | 3 (combinaisons) | all
# --debug pour un run court
```

### Étapes de préparation (normalement gérées par `run_pipeline.sh`)

```bash
# Conversion PARSE → nnU-Net
python scripts/convert_parse_to_nnunet.py --data_dir /chemin/vers/PARSE --dataset_id 100

# Prétraitement nnU-Net
nnUNetv2_plan_and_preprocess -d 100 --verify_dataset_integrity

# Annotations dégradées (évaluation)
python scripts/generate_degraded_dataset.py \
    --dataset_dir "$nnUNet_raw/Dataset100_PARSE" \
    --config configs/experiment_config.yaml --seed 42

# Dataset à annotations dégradées fixes (Dataset101)
python scripts/create_fixed_degraded_dataset.py --dataset_id 101 \
    --source_dir "$nnUNet_raw/Dataset100_PARSE" \
    --output_dir "$nnUNet_raw/Dataset101_PARSE_Fixed" \
    --config configs/experiment_config.yaml --seed 42
nnUNetv2_plan_and_preprocess -d 101 --verify_dataset_integrity
```

---

## 9. Configuration : le fichier à éditer

Tout se règle dans **`configs/experiment_config.yaml`**. Les réglages les plus
utiles :

```yaml
training:
  num_epochs: 250          # ← nombre d'époques (à ajuster via la calibration)
  folds: [0, 1, 2, 3, 4]   # folds de la validation croisée

calibration:
  budget: 1000             # longueur du run de calibration
  milestones: [200, 300, 400, 500, 600, 800, 1000]   # points de mesure

experiment:
  models:                  # la matrice expérimentale (tier A = prioritaire)
    - {name: Model_Star,        trainer: nnUNetTrainer250,       dataset_id: 100, tier: A}
    - {name: Model_Minus_Stoch, trainer: nnUNetTrainerDegraded,  dataset_id: 100, tier: A}
    - {name: Model_Minus_Fixed, trainer: nnUNetTrainer250,       dataset_id: 101, tier: A}

evaluation:
  scenarios:               # contre quelles annotations on évalue
    - {name: GT_star,       degradation: null}        # annotations propres
    - {name: GT_minus_test, degradation: [...]}       # annotations dégradées
```

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

| # | Question | Tier |
|---|----------|------|
| 1 | Qualité du signal vs variabilité (Fixed vs Stoch) | A |
| 3 | Robustesse induite par le bruit stochastique | A |
| 4 | Pénalisation injuste par une GT d'évaluation biaisée | A |
| 6 | Asymétrie train/test de la pénalité | A |
| 2 | Seuils de rupture (grid search) | B |
| 5 | Morpho vs omission (grid search) | B |
| 7 | Spécificité de la robustesse (MorphoOnly / OmissionOnly) | C |

Les outcomes du **tier A** sont produits par défaut. Les tiers **B** et **C**
s'activent avec `--tiers ABC`.
