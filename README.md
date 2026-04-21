# nnU-Net PARSE Pipeline — Étude de robustesse aux dégradations de labels

## Vue d'ensemble

Ce projet utilise **nnU-Net v2** comme backbone de segmentation pour le dataset
PARSE (artères pulmonaires en CT), en se concentrant exclusivement sur l'impact
des dégradations de vérité terrain (GT) sur la performance de segmentation.

### Protocole expérimental

1. **Model_Star** : nnU-Net entraîné sur GT propres (baseline)
2. **Model_Minus** : nnU-Net entraîné sur GT dégradées (morpho + omission on-the-fly)
3. **Cross-évaluation** : chaque modèle est testé sur GT\* et GT-
4. **Grid search** : exploration systématique des paramètres de dégradation

### Structure du projet

```
nnunet_parse_project/
├── README.md
├── requirements.txt
├── setup_env.sh                       # Installation uv + nnU-Net + custom trainer
├── run_pipeline.sh                    # Pipeline complet (SSH / terminal)
├── .env_nnunet                        # Auto-généré par setup_env.sh
├── scripts/
│   ├── convert_parse_to_nnunet.py     # Conversion PARSE → format nnU-Net
│   ├── degradations.py                # Dégradations morpho + omission (NumPy)
│   ├── generate_degraded_dataset.py   # Labels dégradés déterministes pour eval
│   ├── run_experiment.py              # Orchestrateur Python (alternatif au .sh)
│   ├── cross_evaluate.py              # Dice, HD95 — Model_Star vs Model_Minus
│   ├── grid_search.py                 # Grid search des paramètres de dégradation
│   └── aggregate_results.py           # Heatmaps, stats, rapport final
├── custom_trainers/
│   └── nnUNetTrainerDegraded.py       # Trainer avec dégradations on-the-fly
└── configs/
    └── experiment_config.yaml         # Configuration centralisée
```

---

## Quickstart

### 1. Installation

```bash
# macOS (debug / validation du pipeline)
bash setup_env.sh --local

# Serveur GPU A40 (via SSH)
bash setup_env.sh --cluster
```

### 2. Activer l'environnement (chaque session)

```bash
source .env_nnunet
```

### 3. Lancer le pipeline

```bash
# Pipeline complet sur GPU (dans tmux pour les sessions longues)
tmux new -s nnunet
bash run_pipeline.sh /chemin/vers/PARSE

# Mode debug sur macOS (1 fold, rapide)
bash run_pipeline.sh /chemin/vers/PARSE --debug

# Reprendre à une étape spécifique
bash run_pipeline.sh /chemin/vers/PARSE --from train_minus
```

### 4. Étapes individuelles (commandes directes)

```bash
# Conversion
uv run scripts/convert_parse_to_nnunet.py --data_dir /chemin/vers/PARSE

# Prétraitement nnU-Net
nnUNetv2_plan_and_preprocess -d 100 --verify_dataset_integrity

# Entraînement Model_Star (fold 0)
nnUNetv2_train 100 3d_fullres 0 -tr nnUNetTrainer --npz -device mps

# Entraînement Model_Minus (fold 0)
nnUNetv2_train 100 3d_fullres 0 -tr nnUNetTrainerDegraded --npz -device mps

# Prédiction
nnUNetv2_predict -i $nnUNet_raw/Dataset100_PARSE/imagesTs \
    -o predictions/model_star -d 100 -c 3d_fullres -f 0 -device mps

# Cross-évaluation
uv run scripts/cross_evaluate.py --config configs/experiment_config.yaml

# Grid search
uv run scripts/grid_search.py --config configs/experiment_config.yaml --phase 1
```

---

## Compatibilité matérielle

| Plateforme | Device | VRAM | Usage |
|---|---|---|---|
| macOS M1/M2/M3 | `mps` | 24 Go | Debug, validation du pipeline |
| NVIDIA A40 | `cuda` | 35 Go | Expériences complètes |

Le device est **auto-détecté** dans tous les scripts. nnU-Net adapte
automatiquement la taille du patch et le batch size à la mémoire disponible.

---

## Custom trainer : injection des dégradations

Le `nnUNetTrainerDegraded` hérite de `nnUNetTrainer` et override
`train_step()` pour appliquer des dégradations morphologiques et/ou des
omissions de composantes connexes sur les labels **à chaque itération**,
avant le forward pass.

Variantes pré-configurées :

| Trainer | Morpho | Omission |
|---|---|---|
| `nnUNetTrainer` (baseline) | — | — |
| `nnUNetTrainerDegraded` | p=0.3 r=3 | p=0.2 s=150 |
| `nnUNetTrainerDegradedMild` | p=0.1 r=1 | p=0.1 s=50 |
| `nnUNetTrainerDegradedSevere` | p=0.5 r=5 | p=0.5 s=500 |
| `nnUNetTrainerDegradedMorphoOnly` | p=0.3 r=3 | — |
| `nnUNetTrainerDegradedOmissionOnly` | — | p=0.2 s=150 |

Le grid search génère automatiquement des variantes supplémentaires.

---

## Résultats

Les résultats sont stockés dans `results/` :

- `cross_evaluation.csv` : métriques par cas et par scénario
- `grid_search_results.csv` : résultats du grid search
- `figures/` : heatmaps, boxplots, bar charts
- `statistical_tests.csv` : tests de Wilcoxon (significativité)
