# GPU node (cluster) scripts

These scripts run **on the GPU node** (one A40, `gpu-worker-2`), from the repository root,
after `git pull`. The node needs the data in `nnUNet_data/`, the environment from
`setup_env.sh --cluster` (`.env_nnunet`), and `prgt` installed in it once:

```bash
source .env_nnunet
pip install -e . --no-deps      # --no-deps: never upgrade numpy/scipy on the node
```

| Script | Purpose |
|---|---|
| `launch_gpu.sh` | Start the trainings as detached processes (seed replicates, Q3, or study tiers); `--status`, `--stop` |
| `progress.py` | Per-fold progress from the nnU-Net logs: epoch, s/epoch, Dice, estimated end, stalled or duplicate runs |
| `prepare_q3.sh` | Build Dataset103/104 for Q3 (frozen degraded labels, plans and folds of Dataset100) |
| `check_scoring.py` | Check that the node scores exactly like the study (3 cases x 4 references) |
| `launch_cpu_draws.sh` | CPU only: K independent draws of GT- (variance of the noise realisation) |
| `sync_and_report.sh` | From another machine: pull the validation predictions, score and report locally |
| `make_seed_config.py`, `make_q3_config.py`, `list_models.py` | Helpers of `launch_gpu.sh` |

Outputs on the node go to `results_seeds/<model>/` (ledger, `metrics.csv`, logs in
`results_seeds/logs/`). That folder is the node's working directory and is ignored by git.
The scores used by the analyses are copied to `results/cluster_runs/<model>/metrics.csv`
(tracked): from the repository root of the machine that runs the analyses,

```bash
rsync -av --include='*/' --include='metrics.csv' --exclude='*' \
      gpu2:pr_impact_gt/results_seeds/ results/cluster_runs/
```

## 1. Seed replicates (training-seed variance)

A replicate = M0, M1, M2 trained again with another seed (trainers `*_s1`, `*_s2`), same
dataset, same folds. Tier `S` = seed 1, tier `T` = seed 2. Done for tier S, folds 0 and 1
(34 cases), October 2026; result in `results/seed_variance.csv`.

```bash
bash cluster/launch_gpu.sh --tiers S --folds 0 --debug    # 2 epochs; read the printed warning
bash cluster/launch_gpu.sh --tiers S --folds 0 1          # real run
bash cluster/launch_gpu.sh --status
watch -n 60 python cluster/progress.py
bash cluster/launch_gpu.sh --stop                         # stops this node's orchestrators
```

- One process per regime, each with its own ledger and results folder. Rerunning the same
  command resumes from the last checkpoint. `GPUS="0 1"` spreads the processes over GPUs;
  `DA_PROCS=4` lowers the augmentation workers on a node with few cores.
- The launcher refuses to start a model whose orchestrator is running (a double launch on
  2026-09-22 trained every fold twice).
- Scores: `check_scoring.py` must report 12/12 identical rows before replicate scores are
  compared with the study's (it did, 2026-10-12).
- Then, locally: `python analysis/seed_variance.py`.

## 2. Q3: does a network learn an annotation bias? (M3/M4)

M3 is trained on the frozen GT- drift mu=-0.5 (Dataset103), M4 on mu=+0.5 (Dataset104). Their
training labels are exactly the drift-/drift+ evaluation references. The analysis plan was
fixed before the data: docstring of `analysis/q3_learned_bias.py` (primary contrast M4 vs M3).

```bash
bash cluster/prepare_q3.sh --check     # state only
bash cluster/prepare_q3.sh             # ~1 h of CPU, idempotent
GPUS=0 bash cluster/launch_gpu.sh --q3 --folds 0 1
watch -n 60 python cluster/progress.py
```

- `prepare_q3.sh` sets aside any earlier training of 103/104 made with other plans (the July
  2026 attempt used Dataset103's own normalisation) and refuses to run while an M3/M4
  orchestrator is running.
- Duration: standard trainer, 68 s/epoch alone on the A40 (~10 h per fold); M3 and M4 in
  parallel take roughly twice as long each.
- Then copy `results_seeds/M3_*/metrics.csv` and `M4_*` to `results/cluster_runs/` and run
  `python analysis/q3_learned_bias.py` (`--selftest` checks the analysis on synthetic data).

## 3. Independent draws of GT- (CPU)

```bash
bash cluster/launch_cpu_draws.sh --draws 10 --detach   # or --workers N
tail -f results/logs/draws.log                          # end: results/reference_draws_summary.csv
```

Resumable (a finished draw is skipped). The number of workers is bounded by the memory
(~2 GB per worker): a pool that is too wide **hangs without a message** if a worker is killed.

## Moving the node to the restructured repository (October 2026)

The repository was restructured in October 2026 (`scripts/` split into `prgt/`,
`pipeline/`, `cluster/`, ...). On the node, pull only when **no orchestrator is running**
(the running ones call `scripts/...` by path at the end of every fold):

```bash
bash scripts/cluster/launch_gpu.sh --status     # no "[running]" / "[en cours]" line
git fetch origin
B=$(git rev-parse --abbrev-ref HEAD)
# files the new commits add that already exist here, untracked (e.g. results/metrics.csv):
# git would refuse to pull over them, so they are set aside in pre_pull/
git diff --name-only --diff-filter=A HEAD "origin/$B" | while read -r f; do
  if [ -e "$f" ] && ! git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
    mkdir -p "pre_pull/$(dirname "$f")" && mv "$f" "pre_pull/$f"; fi; done
# .env_nnunet is no longer tracked (it holds machine paths): keep the node's copy
cp .env_nnunet ~/.env_nnunet.node && git checkout -- .env_nnunet
git pull
cp ~/.env_nnunet.node .env_nnunet
source .env_nnunet && pip install -e . --no-deps
python cluster/check_scoring.py                 # must still print 12/12 identical rows
```
