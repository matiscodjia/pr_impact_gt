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
python scripts/cluster/progress.py                     # précis : époque, s/époque, dice, fin estimée par pli et par file
bash scripts/cluster/launch_gpu.sh --stop              # arrête tous les processus
```

`progress.py` lit les `training_log_*.txt` de nnU-Net (reprises `--c` comprises) et signale
`BLOQUÉ?` quand un log n'avance plus. `watch -n 60 python scripts/cluster/progress.py` pour
un suivi continu ; M3/M4 : `--datasets 'Dataset10[34]_*' --pattern nnUNetTrainerStd`.
La colonne « actif » teste le PID : elle n'est juste que sur le nœud qui fait tourner le processus.

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

## 3. Q3 -- un réseau apprend-il un biais d'annotation ? (M3/M4, GPU)

M3 est entraîné sur GT⁻ drift μ=−0,5 figée (Dataset103), M4 sur μ=+0,5 (Dataset104). Leurs
labels d'entraînement sont **exactement** les références d'évaluation drift−/drift+ (vérifié
à 85/85 sur le Mac). Plan d'analyse fixé avant les données : docstring de
`analysis/q3_learned_bias.py` (contraste principal M4 contre M3 ; M3 contre M0 serait biaisé,
car M2 montre qu'entraîner sur des bords bruités fait déjà perdre 5,4 % de volume).

**(0) Une fois, depuis le Mac** : copier les références GT⁻ de l'étude sur le cluster. Elles
servent aux labels de M3/M4 et au score contre GT⁻ (sans elles `collect_metrics` ne score que
GT* : c'est ce qui est arrivé au pli 0 des réplicats ; il sera complété au prochain pli).
```bash
rsync -av nnUNet_data/nnUNet_raw/Dataset100_PARSE/labelsTr_GT_minus_{omission,drift_neg,drift_pos} \
      gpu2:pr_impact_gt/nnUNet_data/nnUNet_raw/Dataset100_PARSE/
```
**(1) Sur le nœud qui entraînera M3/M4** (autre worker : vérifier d'abord qu'il voit le même
`~/pr_impact_gt`, sinon rien de ce qui suit ne marche tel quel) :
```bash
git pull
bash scripts/cluster/prepare_q3.sh --check      # état, ne modifie rien
bash scripts/cluster/prepare_q3.sh              # ~1 h de CPU : liens images, labels = GT⁻ (vérifiés
                                                # voxel à voxel), plans ET normalisation de Dataset100,
                                                # plis de M0, prétraitement 3d_fullres ; relançable
GPUS=0 bash scripts/cluster/launch_gpu.sh --q3 --folds 0 1
watch -n 60 python3 scripts/cluster/progress.py  # réplicats + Q3 dans le même tableau
```
- **Un seul lancement.** Le lanceur refuse désormais de relancer un modèle dont
  l'orchestrateur tourne (un double lancement avait fait entraîner chaque pli deux fois).
- **Durée** : entraîneur standard, 68 s/époque seul sur l'A40 (≈ 10 h/pli). M3 et M4 en
  parallèle sur un GPU : environ deux fois plus lent chacun, donc ≈ 1,5-2 jours pour les plis
  0 et 1. Étendre ensuite avec `--folds 2 3 4`.
- **Arrêter Q3 seulement** : `bash scripts/cluster/launch_gpu.sh --stop` *sur ce nœud*. Les
  `.pid` portent le nom du nœud (`pid@nœud`) : un `--stop` n'agit que sur les processus de la
  machine où il est lancé, et ne tue qu'un vrai orchestrateur.
- **Résultats** : `results_seeds/M3_Drift_muMinus/metrics.csv` et `M4_...` (collecte
  automatique après chaque pli, 4 références). Les pousser, puis en local :
  `python analysis/q3_learned_bias.py` (`--selftest` vérifie l'analyse sur données synthétiques).

## Ce qui n'a PAS été testé ici

Ce Mac n'a ni CUDA ni `nnunetv2` : les classes `*_s1/_s2` (`custom_trainers/`) ont été
compilées mais **jamais exécutées**, et `collect_metrics.py` n'a pas été lancé sur un
réplicat. Le test `--debug` de l'étape (a) sert précisément à cela. Testés en local :
génération de la config, listes de modèles, simulation (`--dry-run`) des lanceurs,
`orchestrator.py --dry-run` sur la config générée, `seed_variance.py` (réplicats
synthétiques), `dose_response.py` avec une autre graine.
Q3 (2026-09-25) : `prepare_q3.sh --check/--dry-run` sur une copie isolée (labels = GT⁻ 85/85,
comparaison des plans dans les deux sens), `launch_gpu.sh --q3 --dry-run`, `--stop`/`--status`
avec faux orchestrateurs (local, autre nœud, leurre, ancien `.pid`), `progress.py` sur logs
simulés, `q3_learned_bias.py --selftest` + test nul. **Jamais exécutés ici** : les commandes
nnU-Net de `prepare_q3.sh` (extract_fingerprint, move_plans, preprocess ; options vérifiées
dans les sources de nnunetv2 2.8.1, pas sur la version du cluster) et `setsid`.
