#!/usr/bin/env bash
# Trains the ABIDE voxel models (AAL, Schaefer-200) on the split shared with
# the ABIDE structural arm (503/116/155), so the Ch7 fusion is evaluated on
# the same 155 test subjects. Split: data/abide/abide_split_seed42.json.
# Results: results/ch7/<name>/; logs: logs/<name>.log.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs
SPLIT=data/abide/abide_split_seed42.json

run_cell () {
  local atlas="$1" name="$2"
  echo "================================================================"
  echo "=== [$(date '+%F %T')] START $name  (atlas=$atlas) ==="
  echo "================================================================"
  PYTHONUNBUFFERED=1 DATASET=abide TASK=classification CONNECTIVITY_TYPE=voxel ATLAS="$atlas" \
    SPLIT_FILE="$SPLIT" CHAPTER=7 EXPERIMENT_NAME="$name" KEEP_CACHE=1 \
    stdbuf -oL -eL python3 -u src/train.py 2>&1 | tee "logs/${name}.log"
  echo "=== [$(date '+%F %T')] END $name rc=${PIPESTATUS[0]} ==="
}

run_cell aal_roi_atlas        abide_voxel_aal_sharedsplit
run_cell schaefer_200_7n_yeo  abide_voxel_schaefer_sharedsplit
echo "=================== ALL DONE [$(date '+%F %T')] ==================="
