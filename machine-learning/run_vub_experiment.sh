#!/usr/bin/env bash
#
# V/U/B experiment runner for the SSI transformer.
#
# Trains the three models (voiced-only, unvoiced-only, both) sequentially on the
# same data split, evaluating each under BOTH input regimes (voiced & unvoiced)
# on the main held-out test split plus the membrane held-out set (if present).
#
# Usage:
#   ./run_vub_experiment.sh                    # defaults below
#   EVAL_ONLY=1 ./run_vub_experiment.sh        # skip training, re-eval saved models
#
# Config via environment variables:
#   STUDY_DIR  output folder (default: experiments/vub)
#   MODES      comma list (default: voiced,unvoiced,both)
#   EPOCHS     epochs per run (default: 150)
#   BATCH      batch size (default: 8)
#   SEED       seed (default: 3401)
#   CUDA       cuda device index (default: 0)
#   DATA_PATH  dataset root (default: <repo>/transformer_data)
#   VOC_SAMPLES  number of utterances to vocode (default: 10)
#   VOICED_W     loss weight of the voiced view for mode 'both' (default: 1.0)
#   UNVOICED_W   loss weight of the unvoiced view for mode 'both' (default: 1.0)
#   CURRIC       ramp unvoiced weight 0 -> UNVOICED_W over first N epochs (default: 0)
#   EARLY_STOP   which view's val loss drives early stopping: both|voiced|unvoiced
#   EXTRA      extra args appended to every train command (e.g. "--no-waveform-metrics")
#
# Example (gentle emphasis on the silent view, stop on silent val loss):
#   UNVOICED_W=1.5 CURRIC=40 EARLY_STOP=unvoiced EPOCHS=150 ./run_vub_experiment.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

STUDY_DIR="${STUDY_DIR:-experiments/vub}"
MODES="${MODES:-voiced,unvoiced,both}"
EPOCHS="${EPOCHS:-150}"
BATCH="${BATCH:-8}"
SEED="${SEED:-3401}"
CUDA="${CUDA:-0}"
DATA_PATH="${DATA_PATH:-${ROOT}/transformer_data}"
VOC_SAMPLES="${VOC_SAMPLES:-10}"
EVAL_ONLY="${EVAL_ONLY:-0}"
EXTRA="${EXTRA:-}"

# Optional 'both'-mode weighting / curriculum knobs (passed only to training).
WEIGHT_ARGS=""
[ -n "${VOICED_W:-}" ]   && WEIGHT_ARGS="$WEIGHT_ARGS --both-voiced-weight $VOICED_W"
[ -n "${UNVOICED_W:-}" ] && WEIGHT_ARGS="$WEIGHT_ARGS --both-unvoiced-weight $UNVOICED_W"
[ -n "${CURRIC:-}" ]     && WEIGHT_ARGS="$WEIGHT_ARGS --unvoiced-curriculum-epochs $CURRIC"
[ -n "${EARLY_STOP:-}" ] && WEIGHT_ARGS="$WEIGHT_ARGS --early-stop-source $EARLY_STOP"

# Prefer the project venv if present.
if [ -f "$ROOT/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
  PY="python"
else
  PY="python3"
fi

echo "== V/U/B experiment =="
echo "  study_dir : $STUDY_DIR"
echo "  modes     : $MODES"
echo "  epochs    : $EPOCHS   batch: $BATCH   seed: $SEED   cuda: $CUDA"
echo "  data_path : $DATA_PATH"
echo "  eval_only : $EVAL_ONLY"
[ -n "$WEIGHT_ARGS" ] && echo "  both-weights/curriculum:$WEIGHT_ARGS"

mkdir -p "$STUDY_DIR"

if [ "$EVAL_ONLY" != "0" ]; then
  echo "== eval-only mode: re-evaluating saved models =="
  for MODE in $(echo "$MODES" | tr ',' ' '); do
    echo "--- eval mode=$MODE ---"
    $PY "$ROOT/src/eval_vub.py" \
        --study-dir "$STUDY_DIR" --mode "$MODE" \
        --data-path "$DATA_PATH" \
        --seed "$SEED" --batch-size "$BATCH" --cuda-device "$CUDA" \
        --vocode-samples "$VOC_SAMPLES" \
        $EXTRA
  done
else
  echo "== training + eval (each model also evaluated under voiced AND unvoiced input) =="
  $PY "$ROOT/src/train_vub.py" \
      --study-dir "$STUDY_DIR" --modes "$MODES" \
      --data-path "$DATA_PATH" \
      --epochs "$EPOCHS" --batch-size "$BATCH" --seed "$SEED" \
      --cuda-device "$CUDA" --vocode-samples "$VOC_SAMPLES" \
      $WEIGHT_ARGS $EXTRA
fi

echo
echo "== done. Results =="
echo "  comparison tables :"
echo "    $STUDY_DIR/vub_summary.csv           (main test split, voiced & unvoiced input)"
echo "    $STUDY_DIR/vub_membrane_summary.csv  (membrane held-out, if present)"
echo "  per-run details   :"
echo "    $STUDY_DIR/<mode>/metrics_<mode>_<regime>.json"
echo "    $STUDY_DIR/<mode>/training_history.csv"
echo "    $STUDY_DIR/<mode>/model_final.pt"
