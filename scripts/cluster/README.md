# Lancer sur les workers (ssh direct, A40)

Rien n'est lancé depuis ici : ces scripts s'exécutent **sur le worker**, après `git pull`.
Prérequis déjà satisfaits d'après vous : données dans `nnUNet_data/`, env nnU-Net installé
(`.env_nnunet` produit par `setup_env.sh --cluster`).

## 1. Réplicats de graine (GPU) -- répond à « variance d'entraînement »

Un réplicat = M0, M1, M2 réentraînés avec une autre graine (trainers `*_s1`, `*_s2`),
même dataset, mêmes plis. Tier `S` = graine 1, tier `T` = graine 2.

```bash
git pull
# (a) test mécanique, 2 époques -- LIRE l'avertissement affiché : il faut supprimer ses sorties
bash scripts/cluster/launch_gpu.sh --tiers S --folds 0 --debug
bash scripts/cluster/launch_gpu.sh --status
# (b) lancement réel : commencer par les plis 0 et 1 (34 cas), on étendra ensuite
bash scripts/cluster/launch_gpu.sh --tiers S --folds 0 1
bash scripts/cluster/launch_gpu.sh --status            # GPU, dernière ligne de log, en cours ?
bash scripts/cluster/launch_gpu.sh --stop              # arrête tous les processus
```

- **3 processus en parallèle** (un par régime) sur l'A40 ; chacun a son ledger et son
  `results_seeds/<modèle>/`. Reprenable : relancer la même commande reprend au dernier
  checkpoint. `GPUS="0 1"` répartit sur plusieurs GPU ; `DA_PROCS=4` réduit les workers
  d'augmentation si le worker a peu de coeurs.
- **Étendre** : `--folds 2 3 4`, puis `--tiers T` (graine 2). Les plis sont entrelacés :
  chaque pli terminé donne des résultats exploitables.
- **M3/M4** (biais fixe, papier « future work ») : `--tiers B --study`. M3 repart de zéro
  (aucun checkpoint récupérable), M4 n'a jamais démarré.
- **Durée** : non mesurée. Le seul repère est ~112 s/époque (log de M3, ≈15,6 h/unité de
  500 époques). Partager l'A40 entre 3 processus ralentit chacun ; à mesurer sur la première
  heure avant de planifier.

Les métriques des réplicats sont calculées sur le worker (l'orchestrateur appelle
`collect_metrics.py` après chaque pli) dans `results_seeds/<modèle>/metrics.csv`.
Ne poussez que ces CSV (petits) : `git add results_seeds/*/metrics.csv`. En local :

```bash
git pull
python analysis/seed_variance.py        # -> results/seed_variance.csv
```

## 2. Tirages multiples de GT- (CPU) -- répond à « une seule réalisation du bruit »

```bash
bash scripts/cluster/launch_cpu_draws.sh --draws 10 --detach   # ou --workers N
tail -f results/logs/draws.log
# fin : results/reference_draws_summary.csv  (pousser les CSV de results/)
```

Reprenable (un tirage terminé est sauté). Le nombre de workers est borné par la RAM
(~2 Go/worker) : un pool trop large **se bloque sans message** si un worker est tué.

## Ce qui n'a PAS été testé ici

Ce Mac n'a ni CUDA ni `nnunetv2` : les classes `*_s1/_s2` (`custom_trainers/`) ont été
compilées mais **jamais exécutées**, et `collect_metrics.py` n'a pas été lancé sur un
réplicat. Le test `--debug` de l'étape (a) sert précisément à cela. Testés en local :
génération de la config, listes de modèles, simulation (`--dry-run`) des lanceurs,
`orchestrator.py --dry-run` sur la config générée, `seed_variance.py` (réplicats
synthétiques), `dose_response.py` avec une autre graine.
