#!/usr/bin/env bash
# ============================================================================
# prepare_q3.sh -- prépare l'entraînement Q3 : un réseau apprend-il un biais d'annotation ?
#   M3 : labels d'entraînement = GT⁻ drift μ=−0,5 (sous-segmentation), Dataset103
#   M4 : labels d'entraînement = GT⁻ drift μ=+0,5 (sur-segmentation), Dataset104
#
# À exécuter SUR le nœud qui entraînera M3/M4, racine du repo, après `git pull` :
#     bash scripts/cluster/prepare_q3.sh --check     # état seulement, ne modifie rien
#     bash scripts/cluster/prepare_q3.sh             # tout (~1 h de CPU), idempotent, relançable
#     bash scripts/cluster/prepare_q3.sh --dry-run   # données brutes (léger) faites, commandes
#                                                    # nnU-Net seulement affichées
#
# Ce que le script garantit (sans quoi Q3 ne répond pas à sa question) :
#  1. labelsTr de Dataset103/104 = EXACTEMENT les références d'évaluation GT⁻ drift−/drift+,
#     copiées depuis Dataset100/labelsTr_GT_minus_* et vérifiées voxel à voxel : le même
#     « annotateur biaisé » à l'entraînement et à l'évaluation (vérifié aussi sur le Mac :
#     create_fixed_degraded_dataset.py, graine 42 + indice, redonne ces fichiers à 85/85).
#  2. plans nnU-Net = ceux de Dataset100 (move_plans), normalisation des intensités comprise.
#     Un plan_and_preprocess propre à 103/104 la calculerait sous le masque BIAISÉ : M3/M4
#     différeraient alors de M0 par autre chose que leurs labels.
#  3. mêmes plis que M0 (splits_final.json de Dataset100).
#  4. dataset.json dans nnUNet_preprocessed : c'est la planification qui l'y copie, étape
#     sautée ici (plans repris) ; sans lui nnUNetv2_train échoue au démarrage.
# Variables : NP=8 (processus nnU-Net), PYTHON=python
# ============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
MODE="run"
case "${1:-}" in
  --check) MODE="check" ;; --dry-run) MODE="dry" ;; "") ;;
  *) echo "argument inconnu: $1"; exit 1 ;;
esac
[[ -f .env_nnunet ]] && source .env_nnunet
PY="${PYTHON:-python}"; NP="${NP:-8}"
RAW="${nnUNet_raw:-nnUNet_data/nnUNet_raw}"
PRE="${nnUNet_preprocessed:-nnUNet_data/nnUNet_preprocessed}"
RES="${nnUNet_results:-nnUNet_data/nnUNet_results}"
SRC="$RAW/Dataset100_PARSE"; SRC_PRE="$PRE/Dataset100_PARSE"
IDS=(103 104); NAMES=(Dataset103_PARSE_DriftMuMinus Dataset104_PARSE_DriftMuPlus); REFS=(drift_neg drift_pos)

run()  { if [[ "$MODE" == dry ]]; then echo "    [dry-run] $*"; else echo "    \$ $*"; "$@"; fi; }
fail() { echo "ÉCHEC : $*" >&2; exit 1; }
count() { find "$1" -maxdepth 1 -name "$2" 2>/dev/null | wc -l | tr -d ' '; }

# ---- 0. nœud ------------------------------------------------------------------
echo "== 0. nœud $(hostname) | $(nproc 2>/dev/null || sysctl -n hw.ncpu) cœurs | mode $MODE"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader | sed 's/^/   GPU /'
free_gb=$(df -Pk "$PRE" | awk 'NR==2 {print int($4/1048576)}')
echo "   espace libre sous $PRE : ${free_gb} Go"
[[ "$free_gb" -ge 40 ]] || echo "   !! moins de 40 Go : le prétraitement de 103 + 104 risque de manquer de place"
if pgrep -f -- '^[^ ]*python[0-9.]* +([^ ]*/)?scripts/orchestrator\.py .*--results_dir results_seeds/M[34]_' >/dev/null; then
  fail "un orchestrateur M3/M4 tourne déjà : ne pas préparer pendant l'entraînement"
fi
[[ "$MODE" != run ]] || command -v nnUNetv2_preprocess >/dev/null || fail "nnUNetv2 introuvable (source .env_nnunet ?)"
N=$(count "$SRC/labelsTr" '*.nii.gz'); [[ "$N" -gt 0 ]] || fail "$SRC/labelsTr vide"

# ---- 1. références d'évaluation ----------------------------------------------------
echo "== 1. références GT⁻ de l'étude (copiées depuis le Mac, graine 42)"
missing=0
for s in omission drift_neg drift_pos; do
  n=$(count "$SRC/labelsTr_GT_minus_$s" '*.nii.gz'); echo "   labelsTr_GT_minus_$s : $n/$N"
  [[ "$n" == "$N" ]] || missing=1
done
if [[ "$missing" == 1 ]]; then
  cat <<MSG
   Manquantes : elles servent aux labels de M3/M4 ET au calcul des scores contre GT⁻
   (sans elles, collect_metrics ne score que GT*). Depuis le Mac, racine du repo :
     rsync -av nnUNet_data/nnUNet_raw/Dataset100_PARSE/labelsTr_GT_minus_{omission,drift_neg,drift_pos} \\
           gpu2:pr_impact_gt/nnUNet_data/nnUNet_raw/Dataset100_PARSE/
MSG
  exit 1
fi

# ---- 2. données brutes 103/104 : images = liens vers Dataset100, labels = GT⁻ ------------
echo "== 2. données brutes"
for i in 0 1; do
  name=${NAMES[$i]}; ref=${REFS[$i]}; d="$RAW/$name"
  if [[ "$MODE" == check ]]; then
    echo "   $name : imagesTr $(count "$d/imagesTr" '*.nii.gz')/$N, labelsTr $(count "$d/labelsTr" '*.nii.gz')/$N"
    continue
  fi
  mkdir -p "$d/imagesTr" "$d/labelsTr"
  src_img="$(cd "$SRC/imagesTr" && pwd -P)"
  for f in "$src_img"/*.nii.gz; do ln -sfn "$f" "$d/imagesTr/$(basename "$f")"; done
  for f in "$SRC/labelsTr_GT_minus_$ref"/*.nii.gz; do
    t="$d/labelsTr/$(basename "$f")"; [[ -e "$t" ]] || cp -p "$f" "$t"
  done
  "$PY" - "$SRC/dataset.json" "$d/dataset.json" "$name" "$ref" <<'EOF'
import json, sys
src, dst, name, ref = sys.argv[1:]
d = json.load(open(src))
d["name"] = name
d["description"] = f"PARSE ; labels d'entraînement = labelsTr_GT_minus_{ref} de Dataset100 (scripts/cluster/prepare_q3.sh)"
json.dump(d, open(dst, "w"), indent=2)
EOF
  echo "   $name : imagesTr $(count "$d/imagesTr" '*.nii.gz')/$N liens, labelsTr $(count "$d/labelsTr" '*.nii.gz')/$N"
done

# ---- 3. vérification voxel à voxel ------------------------------------------------------
echo "== 3. labels d'entraînement = références d'évaluation ? (voxel à voxel, quelques minutes)"
if ! "$PY" - "$SRC" "$RAW" <<'EOF'
import glob, os, sys
import nibabel as nib
import numpy as np
src, raw = sys.argv[1:]
cases = sorted(os.path.basename(f) for f in glob.glob(f"{src}/labelsTr/*.nii.gz"))
ok = True
for name, ref in (("Dataset103_PARSE_DriftMuMinus", "drift_neg"), ("Dataset104_PARSE_DriftMuPlus", "drift_pos")):
    same = absent = 0
    for c in cases:
        lab = f"{raw}/{name}/labelsTr/{c}"
        img = f"{raw}/{name}/imagesTr/{c[:-7]}_0000.nii.gz"
        if not (os.path.exists(lab) and os.path.exists(img)):
            absent += 1
            continue
        a = np.asanyarray(nib.load(lab).dataobj) > 0
        b = np.asanyarray(nib.load(f"{src}/labelsTr_GT_minus_{ref}/{c}").dataobj) > 0
        same += int(a.shape == b.shape and bool((a == b).all()))
    print(f"   {name} : labels = GT⁻ {ref} sur {same}/{len(cases)} cas ({absent} image/label absent)")
    ok &= same == len(cases)
sys.exit(0 if ok else 1)
EOF
then
  [[ "$MODE" == check ]] && echo "   -> à préparer (lancer sans --check)" || \
    fail "labels différents des références : supprimer $RAW/Dataset10{3,4}_*/labelsTr et relancer"
fi

# ---- 4. plans de Dataset100, dataset.json, plis, prétraitement 3d_fullres ------------------
plans_equal() {   # 0 si $1/nnUNetPlans.json = plans de Dataset100 (hors nom du dataset)
  "$PY" - "$SRC_PRE/nnUNetPlans.json" "$1/nnUNetPlans.json" 2>/dev/null <<'EOF'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:])
for k in ("dataset_name", "plans_name", "image_reader_writer"):
    a.pop(k, None); b.pop(k, None)
sys.exit(0 if a == b else 1)
EOF
}
echo "== 4. plans, plis, prétraitement"
for i in 0 1; do
  id=${IDS[$i]}; name=${NAMES[$i]}; tp="$PRE/$name"
  if [[ -f "$tp/.q3_ready" ]] && plans_equal "$tp" && cmp -s "$SRC_PRE/splits_final.json" "$tp/splits_final.json"; then
    echo "   $name : prêt ($(cat "$tp/.q3_ready"))"; continue
  fi
  if [[ "$MODE" == check ]]; then
    echo "   $name : À PRÉPARER (plans $(plans_equal "$tp" && echo "= Dataset100" || echo "absents ou différents"))"
    continue
  fi
  echo "   $name :"
  run nnUNetv2_extract_fingerprint -d "$id" -np "$NP"      # crée le dossier + dataset_fingerprint.json
  run nnUNetv2_move_plans_between_datasets -s 100 -t "$id" -sp nnUNetPlans -tp nnUNetPlans
  [[ "$MODE" == dry ]] || plans_equal "$tp" || fail "plans de $name différents de ceux de Dataset100"
  run cp "$RAW/$name/dataset.json" "$tp/dataset.json"
  run cp "$SRC_PRE/splits_final.json" "$tp/splits_final.json"
  run nnUNetv2_preprocess -d "$id" -plans_name nnUNetPlans -c 3d_fullres -np "$NP"
  if [[ "$MODE" == run ]]; then
    n=$(count "$tp/nnUNetPlans_3d_fullres" '*.pkl')
    [[ "$n" == "$N" ]] || fail "prétraitement de $name incomplet : $n/$N cas"
    echo "$(date '+%F %T') commit $(git rev-parse --short HEAD)" > "$tp/.q3_ready"
    echo "   $name : prêt"
  fi
done

# ---- 5. essais avortés de juillet (dossiers de pli sans checkpoint) ----------------------
echo "== 5. anciennes sorties d'entraînement"
for i in 0 1; do
  td="$RES/${NAMES[$i]}/nnUNetTrainerStd__nnUNetPlans__3d_fullres"
  for fd in "$td"/fold_*; do
    [[ -d "$fd" ]] || continue
    if compgen -G "$fd/checkpoint_*.pth" >/dev/null; then
      echo "   $fd : checkpoint présent, repris tel quel (--c)"; continue
    fi
    dst="$td/aborted_$(date +%Y%m%d)_$(basename "$fd")"
    if [[ "$MODE" == run ]]; then mv "$fd" "$dst"; echo "   $(basename "$fd") -> $(basename "$dst") (sans checkpoint)"
    else echo "   $(basename "$fd") : sans checkpoint -> sera déplacé vers $(basename "$dst")"; fi
  done
done

[[ "$MODE" == run ]] || { echo; echo "($MODE : rien de lourd n'a été lancé)"; exit 0; }
cat <<MSG

Q3 prêt. Lancement (un seul lancement ; GPU au choix, ici 0) :
    GPUS=0 bash scripts/cluster/launch_gpu.sh --q3 --folds 0 1
Suivi :
    watch -n 60 python3 scripts/cluster/progress.py
MSG
