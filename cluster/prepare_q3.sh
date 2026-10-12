#!/usr/bin/env bash
# ============================================================================
# prepare_q3.sh -- prepare the Q3 trainings: does a network learn an annotation bias?
#   M3: training labels = GT- drift mu=-0.5 (under-segmentation), Dataset103
#   M4: training labels = GT- drift mu=+0.5 (over-segmentation),  Dataset104
#
# Run ON the node that will train M3/M4, from the repository root:
#     bash cluster/prepare_q3.sh --check     # state only, changes nothing
#     bash cluster/prepare_q3.sh             # everything (~1 h of CPU), idempotent, rerunnable
#     bash cluster/prepare_q3.sh --dry-run   # raw data (light) done, nnU-Net commands only printed
#
# What the script guarantees (without it, Q3 does not answer its question):
#  1. labelsTr of Dataset103/104 = EXACTLY the evaluation references GT- drift-/drift+,
#     copied from Dataset100/labelsTr_GT_minus_* and checked voxel by voxel: the same
#     "biased annotator" at training and at evaluation (also checked on the Mac:
#     pipeline/make_fixed_training_sets.py, seed 42 + index, gives these files 85/85).
#  2. nnU-Net plans = those of Dataset100 (move_plans), intensity normalisation included.
#     A plan_and_preprocess of 103/104 would compute it under the BIASED mask, and M3/M4
#     would differ from M0 by more than their labels.
#  3. same folds as M0 (splits_final.json of Dataset100).
#  4. dataset.json in nnUNet_preprocessed: planning normally copies it there, a step skipped
#     here (plans reused); without it nnUNetv2_train fails at start-up.
#  5. no training of 103/104 made with other plans is resumed (see step 5).
# Variables: NP=8 (nnU-Net processes), PYTHON=python
# ============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
MODE="run"
case "${1:-}" in
  --check) MODE="check" ;; --dry-run) MODE="dry" ;; "") ;;
  *) echo "unknown argument: $1"; exit 1 ;;
esac
[[ -f .env_nnunet ]] && source .env_nnunet
PY="${PYTHON:-python}"; NP="${NP:-8}"
RAW="${nnUNet_raw:-nnUNet_data/nnUNet_raw}"
PRE="${nnUNet_preprocessed:-nnUNet_data/nnUNet_preprocessed}"
RES="${nnUNet_results:-nnUNet_data/nnUNet_results}"
SRC="$RAW/Dataset100_PARSE"; SRC_PRE="$PRE/Dataset100_PARSE"
IDS=(103 104); NAMES=(Dataset103_PARSE_DriftMuMinus Dataset104_PARSE_DriftMuPlus); REFS=(drift_neg drift_pos)

run()  { if [[ "$MODE" == dry ]]; then echo "    [dry-run] $*"; else echo "    \$ $*"; "$@"; fi; }
fail() { echo "FAILED: $*" >&2; exit 1; }
count() { find "$1" -maxdepth 1 -name "$2" 2>/dev/null | wc -l | tr -d ' '; }

# ---- 0. node ------------------------------------------------------------------
echo "== 0. node $(hostname) | $(nproc 2>/dev/null || sysctl -n hw.ncpu) cores | mode $MODE"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader | sed 's/^/   GPU /'
free_gb=$(df -Pk "$PRE" | awk 'NR==2 {print int($4/1048576)}')
echo "   free space under $PRE: ${free_gb} GB"
[[ "$free_gb" -ge 40 ]] || echo "   !! less than 40 GB: preprocessing 103 + 104 may run out of space"
if pgrep -f -- '^[^ ]*python[0-9.]* +([^ ]*/)?(pipeline|scripts)/orchestrator\.py .*--results_dir results_seeds/M[34]_' >/dev/null; then
  fail "an M3/M4 orchestrator is already running: do not prepare during training"
fi
[[ "$MODE" != run ]] || command -v nnUNetv2_preprocess >/dev/null || fail "nnUNetv2 not found (source .env_nnunet?)"
N=$(count "$SRC/labelsTr" '*.nii.gz'); [[ "$N" -gt 0 ]] || fail "$SRC/labelsTr is empty"

# ---- 1. evaluation references --------------------------------------------------------
echo "== 1. GT- references of the study (seed 42)"
missing=0
for s in omission drift_neg drift_pos; do
  n=$(count "$SRC/labelsTr_GT_minus_$s" '*.nii.gz'); echo "   labelsTr_GT_minus_$s : $n/$N"
  [[ "$n" == "$N" ]] || missing=1
done
if [[ "$missing" == 1 ]]; then
  cat <<MSG
   Missing: they are the labels of M3/M4 AND the references of the GT- scores (without
   them, collect_metrics only scores GT*). From the machine that has them, repository root:
     rsync -av nnUNet_data/nnUNet_raw/Dataset100_PARSE/labelsTr_GT_minus_{omission,drift_neg,drift_pos} \\
           gpu2:pr_impact_gt/nnUNet_data/nnUNet_raw/Dataset100_PARSE/
MSG
  exit 1
fi

# ---- 2. raw data 103/104: images = links to Dataset100, labels = GT- ---------------------
echo "== 2. raw data"
for i in 0 1; do
  name=${NAMES[$i]}; ref=${REFS[$i]}; d="$RAW/$name"
  if [[ "$MODE" == check ]]; then
    echo "   $name: imagesTr $(count "$d/imagesTr" '*.nii.gz')/$N, labelsTr $(count "$d/labelsTr" '*.nii.gz')/$N"
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
d["description"] = f"PARSE; training labels = labelsTr_GT_minus_{ref} of Dataset100 (cluster/prepare_q3.sh)"
json.dump(d, open(dst, "w"), indent=2)
EOF
  echo "   $name: imagesTr $(count "$d/imagesTr" '*.nii.gz')/$N links, labelsTr $(count "$d/labelsTr" '*.nii.gz')/$N"
done

# ---- 3. voxel-by-voxel check ------------------------------------------------------------
echo "== 3. training labels = evaluation references? (voxel by voxel, a few minutes)"
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
    print(f"   {name}: labels = GT- {ref} in {same}/{len(cases)} cases ({absent} image/label missing)")
    ok &= same == len(cases)
sys.exit(0 if ok else 1)
EOF
then
  [[ "$MODE" == check ]] && echo "   -> to prepare (run without --check)" || \
    fail "labels differ from the references: delete $RAW/Dataset10{3,4}_*/labelsTr and rerun"
fi

# ---- 4. plans of Dataset100, dataset.json, folds, 3d_fullres preprocessing ----------------
plans_equal() {   # 0 if the plans file $1 = the plans of Dataset100 (dataset name aside)
  "$PY" - "$SRC_PRE/nnUNetPlans.json" "$1" 2>/dev/null <<'EOF'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:])
for k in ("dataset_name", "plans_name", "image_reader_writer"):
    a.pop(k, None); b.pop(k, None)
sys.exit(0 if a == b else 1)
EOF
}
echo "== 4. plans, folds, preprocessing"
for i in 0 1; do
  id=${IDS[$i]}; name=${NAMES[$i]}; tp="$PRE/$name"
  if [[ -f "$tp/.q3_ready" ]] && plans_equal "$tp/nnUNetPlans.json" && cmp -s "$SRC_PRE/splits_final.json" "$tp/splits_final.json"; then
    echo "   $name: ready ($(cat "$tp/.q3_ready"))"; continue
  fi
  if [[ "$MODE" == check ]]; then
    echo "   $name: TO PREPARE (plans $(plans_equal "$tp/nnUNetPlans.json" && echo "= Dataset100" || echo "missing or different"))"
    continue
  fi
  echo "   $name:"
  run nnUNetv2_extract_fingerprint -d "$id" -np "$NP"      # creates the folder + dataset_fingerprint.json
  run nnUNetv2_move_plans_between_datasets -s 100 -t "$id" -sp nnUNetPlans -tp nnUNetPlans
  [[ "$MODE" == dry ]] || plans_equal "$tp/nnUNetPlans.json" || fail "plans of $name differ from those of Dataset100"
  run cp "$RAW/$name/dataset.json" "$tp/dataset.json"
  run cp "$SRC_PRE/splits_final.json" "$tp/splits_final.json"
  run nnUNetv2_preprocess -d "$id" -plans_name nnUNetPlans -c 3d_fullres -np "$NP"
  if [[ "$MODE" == run ]]; then
    n=$(count "$tp/nnUNetPlans_3d_fullres" '*.pkl')
    [[ "$n" == "$N" ]] || fail "preprocessing of $name incomplete: $n/$N cases"
    echo "$(date '+%F %T') commit $(git rev-parse --short HEAD)" > "$tp/.q3_ready"
    echo "   $name: ready"
  fi
done

# ---- 5. earlier training outputs ----------------------------------------------------------
# nnU-Net copies the plans of a training to <trainer>/plans.json. Plans = Dataset100: output
# of a Q3 launch, resumed (--c) if the fold has a checkpoint. Otherwise (the July 2026 attempt,
# with plans computed on Dataset103 itself = different CT normalisation): the WHOLE folder is
# set aside, checkpoints included; resuming it would mix two normalisations in M3.
echo "== 5. earlier training outputs"
for i in 0 1; do
  td="$RES/${NAMES[$i]}/nnUNetTrainerStd__nnUNetPlans__3d_fullres"
  [[ -d "$td" ]] || continue
  if ! plans_equal "$td/plans.json"; then
    dst="$RES/${NAMES[$i]}/aborted_$(date +%Y%m%d)_$(basename "$td")"
    if [[ "$MODE" == run ]]; then mv "$td" "$dst"; echo "   $(basename "$td") -> $(basename "$dst") (plans != Dataset100)"
    else echo "   ${NAMES[$i]}: training with plans != Dataset100 -> will be moved to $(basename "$dst")"; fi
    continue
  fi
  for fd in "$td"/fold_*; do
    [[ -d "$fd" ]] || continue
    if compgen -G "$fd/checkpoint_*.pth" >/dev/null; then
      echo "   $fd: checkpoint with the plans of Dataset100, resumed (--c)"; continue
    fi
    dst="$td/aborted_$(date +%Y%m%d)_$(basename "$fd")"
    if [[ "$MODE" == run ]]; then mv "$fd" "$dst"; echo "   $(basename "$fd") -> $(basename "$dst") (no checkpoint)"
    else echo "   $(basename "$fd"): no checkpoint -> will be moved to $(basename "$dst")"; fi
  done
done

[[ "$MODE" == run ]] || { echo; echo "($MODE: nothing heavy was run)"; exit 0; }
cat <<MSG

Q3 ready. Launch (once; any GPU, here 0):
    GPUS=0 bash cluster/launch_gpu.sh --q3 --folds 0 1
Follow:
    watch -n 60 python3 cluster/progress.py
MSG
