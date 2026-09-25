#!/usr/bin/env bash
# ============================================================================
# launch_gpu.sh -- lance les entraînements nnU-Net (réplicats de graine, ou tiers de l'étude)
# sur le worker, en processus détachés (survivent à la déconnexion ssh).
#
# À exécuter SUR le worker, depuis la racine du repo, après `git pull` :
#     bash scripts/cluster/launch_gpu.sh --debug                 # 2 époques, valide la mécanique
#     bash scripts/cluster/launch_gpu.sh --tiers S               # graine 1, 3 régimes, 5 plis
#     bash scripts/cluster/launch_gpu.sh --tiers S --folds 0 1   # seulement les plis 0 et 1
#     bash scripts/cluster/launch_gpu.sh --tiers T               # graine 2
#     bash scripts/cluster/launch_gpu.sh --q3 --folds 0 1        # Q3 : M3/M4 (après prepare_q3.sh)
#     bash scripts/cluster/launch_gpu.sh --tiers B --study       # M3/M4 avec la config complète (déconseillé :
#                                                                 #  re-score M0-M2 à chaque pli)
#     bash scripts/cluster/launch_gpu.sh --status | --stop
#
# Un processus par modèle, chacun avec SON dossier de résultats (ledger.json et metrics.csv
# séparés) : pas de course d'écriture entre processus, et rien de ce qui est produit ici ne
# touche results/ (donc pas de conflit git avec le master local).
# Sorties : results_seeds/<modèle>/{ledger.json,metrics.csv,...} et results_seeds/logs/<modèle>.log
#
# Variables : GPUS="0"  (liste de GPU, répartis en tourniquet entre processus)
#             DA_PROCS=6 (nnUNet_n_proc_DA par processus ; réduire si le worker a peu de coeurs)
# ============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
ROOT="$(pwd)"

TIERS="S"; FOLDS=(); DEBUG=0; STUDY=0; Q3=0; ACTION="launch"; DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tiers)  TIERS="$2"; shift 2 ;;
    --folds)  shift; FOLDS=(); while [[ $# -gt 0 && "$1" != --* ]]; do FOLDS+=("$1"); shift; done ;;
    --debug)  DEBUG=1; shift ;;
    --study)  STUDY=1; shift ;;      # utilise experiment_config.yaml (M0-M4) au lieu des réplicats
    --q3)     Q3=1; TIERS="B"; shift ;;   # M3/M4 seuls (configs/generated_q3_config.yaml)
    --status) ACTION="status"; shift ;;
    --stop)   ACTION="stop"; shift ;;
    --dry-run) DRY=1; shift ;;
    *) echo "argument inconnu: $1"; exit 1 ;;
  esac
done

OUT="results_seeds"; mkdir -p "$OUT/logs"
[[ -f .env_nnunet ]] && source .env_nnunet          # nnUNet_raw/preprocessed/results + venv
PY="${PYTHON:-python}"

# ---- stop / status -----------------------------------------------------------
# Un orchestrateur = un python qui EXÉCUTE scripts/orchestrator.py (motif ancré : un shell dont
# la commande mentionne simplement ce nom, ou un grep, ne doit jamais être pris pour lui).
ORCH_RE='^[^ ]*python[0-9.]* +([^ ]*/)?scripts/orchestrator\.py '
running_orchestrators() {   # PID de tout orchestrateur LOCAL écrivant sous $OUT/ (= son groupe, via setsid)
  pgrep -f -- "${ORCH_RE}.*--results_dir $OUT/${1:-}" || true
}
HOST="$(hostname)"
# Fichier .pid = "pid@nœud" : avec un /home partagé entre workers, results_seeds/logs/ est
# commun, et un PID n'a de sens que sur le nœud qui l'a créé (ancien format "pid" = ce nœud).
pid_of()  { cut -d@ -f1 < "$1"; }
host_of() { local h; h=$(cut -s -d@ -f2 < "$1"); echo "${h:-$HOST}"; }
is_orch() { ps -o args= -p "$1" 2>/dev/null | grep -Eq "$ORCH_RE"; }   # PID réutilisé ? ancien .pid ?
if [[ "$ACTION" == "stop" ]]; then
  # .pid (de CE nœud) ET recherche par ligne de commande : un double lancement écrase les .pid,
  # et les orchestrateurs du premier lancement survivraient à un --stop fondé sur les seuls .pid.
  pids=$(running_orchestrators)
  for f in "$OUT"/logs/*.pid; do
    [[ -f "$f" ]] || continue
    if [[ "$(host_of "$f")" != "$HOST" ]]; then
      echo "$(basename "$f" .pid) tourne sur $(host_of "$f") : --stop à lancer là-bas"; continue
    fi
    pids="$pids $(pid_of "$f")"; rm -f "$f"
  done
  for pid in $(echo $pids | tr ' ' '\n' | sort -u); do
    is_orch "$pid" || { echo "$pid n'est pas un orchestrateur sur $HOST : ignoré"; continue; }
    { kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null; } && echo "arrêté $pid" || true
  done
  exit 0
fi
if [[ "$ACTION" == "status" ]]; then
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
  for f in "$OUT"/logs/*.log; do [[ -f "$f" ]] || continue
    echo "== $(basename "$f" .log)"; tail -n 2 "$f" | cut -c1-160
    pf="${f%.log}.pid"; [[ -f "$pf" ]] || continue
    if [[ "$(host_of "$pf")" != "$HOST" ]]; then echo "   [sur $(host_of "$pf")]"
    else is_orch "$(pid_of "$pf")" && echo "   [en cours]" || echo "   [terminé/arrêté]"; fi
  done
  exit 0
fi

# ---- pré-vol ------------------------------------------------------------------
: "${nnUNet_raw:?nnUNet_raw non defini - faire: source .env_nnunet}"
[[ -d "$nnUNet_raw/Dataset100_PARSE" ]] || { echo "Dataset100_PARSE introuvable dans $nnUNet_raw"; exit 1; }
[[ "$DRY" == 1 ]] || command -v nvidia-smi >/dev/null || { echo "nvidia-smi absent : pas un worker GPU"; exit 1; }

# Les trainers sont COPIÉS dans l'arbre nnU-Net (cf. setup_env.sh) : à refaire après chaque
# modification de custom_trainers/ (c'est le cas ici : nouveaux réplicats _s1/_s2).
if [[ "$DRY" == 0 ]]; then
  TDIR="$($PY -c "import nnunetv2,os;print(os.path.join(nnunetv2.__path__[0],'training','nnUNetTrainer'))")"
  cp custom_trainers/nnUNetTrainerDegraded.py scripts/degradations.py "$TDIR/"
  echo "trainers installés dans $TDIR"
  $PY -c "from nnunetv2.training.nnUNetTrainer.nnUNetTrainerDegraded import nnUNetTrainerStd_s1, nnUNetTrainerDriftMu0_s2; print('import des réplicats: OK')"
fi

if [[ "$Q3" == 1 ]]; then
  # Les plans/données de 103-104 doivent venir de prepare_q3.sh (plans de Dataset100, labels = GT⁻).
  for n in Dataset103_PARSE_DriftMuMinus Dataset104_PARSE_DriftMuPlus; do
    [[ -f "${nnUNet_preprocessed:-nnUNet_data/nnUNet_preprocessed}/$n/.q3_ready" ]] && continue
    [[ "$DRY" == 1 ]] && { echo "(dry-run) $n pas encore préparé"; continue; }
    echo "$n non préparé : lancer d'abord  bash scripts/cluster/prepare_q3.sh"; exit 1
  done
  $PY scripts/cluster/make_q3_config.py; CFG="configs/generated_q3_config.yaml"
elif [[ "$STUDY" == 1 ]]; then CFG="configs/experiment_config.yaml"
else $PY scripts/cluster/make_seed_config.py; CFG="configs/generated_seeds_config.yaml"; fi
[[ "$DEBUG" == 1 ]] && export DEBUG_PIPELINE=1
if [[ "$DEBUG" == 1 ]]; then
  cat <<'MSG'
!! --debug entraine 2 epoques et ECRIT checkpoint_final.pth + validation/ dans
!! nnUNet_results/Dataset100_PARSE/<trainer>__nnUNetPlans__3d_fullres/fold_<k>/.
!! L'orchestrateur y verrait une unite "terminee" et SAUTERAIT le vrai entrainement.
!! Apres le test, supprimer ces dossiers ET les ledgers de test, avant le lancement reel :
!!   rm -rf nnUNet_data/nnUNet_results/Dataset100_PARSE/*_s1__nnUNetPlans__3d_fullres/fold_0 results_seeds
MSG
fi
export nnUNet_n_proc_DA="${DA_PROCS:-6}"

# modèles de la config appartenant aux tiers demandés
MODELS=(); while IFS= read -r line; do MODELS+=("$line"); done < <($PY scripts/cluster/list_models.py "$CFG" "$TIERS")
[[ ${#MODELS[@]} -gt 0 ]] || { echo "aucun modèle pour les tiers '$TIERS' dans $CFG"; exit 1; }
IFS=' ' read -r -a GPU_LIST <<< "${GPUS:-0}"

i=0; launched=0
for m in "${MODELS[@]}"; do
  # Garde-fou : relancer pendant que ça tourne entraînerait chaque pli DEUX fois dans le même
  # dossier (checkpoints et validation/ écrasés, GPU partagé en deux). Arrivé le 22/09.
  if [[ -n "$(running_orchestrators "$m( |$)")" ]]; then
    echo "[$m] déjà en cours (pid $(running_orchestrators "$m( |$)" | tr '\n' ' ')) -- non relancé"
    continue
  fi
  gpu="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"; i=$((i+1))
  cmd=( "$PY" scripts/orchestrator.py --config "$CFG" --results_dir "$OUT/$m" --tiers "$TIERS"
        --models "$m" --no-progress )
  # réplicats : pas de .npz (softmax) -- ~2/3 du poids d'un pli, inutile pour nos analyses.
  # KEEP_NPZ=1 pour les garder ; le mode --study (M0-M4) reste inchangé.
  [[ "$STUDY" == 0 && "${KEEP_NPZ:-0}" != 1 ]] && cmd+=( --no-npz )
  [[ ${#FOLDS[@]} -gt 0 ]] && cmd+=( --folds "${FOLDS[@]}" )
  [[ "$DEBUG" == 1 ]] && cmd+=( --debug )
  echo "[$m] GPU $gpu : ${cmd[*]}"
  [[ "$DRY" == 1 ]] && continue
  mkdir -p "$OUT/$m"
  # setsid : nouveau groupe de processus => survit au logout ssh ; --stop tue tout le groupe.
  # PYTHONUNBUFFERED : sinon le log du modèle (un fichier) reste vide des heures (tampon Python).
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 setsid nohup "${cmd[@]}" >> "$OUT/logs/$m.log" 2>&1 < /dev/null &
  echo "$!@$HOST" > "$OUT/logs/$m.pid"; launched=$((launched+1))
done
[[ "$DRY" == 1 ]] && { echo; echo "dry-run : rien lancé."; exit 0; }
echo; echo "lancé $launched processus. Suivi : bash scripts/cluster/launch_gpu.sh --status"
