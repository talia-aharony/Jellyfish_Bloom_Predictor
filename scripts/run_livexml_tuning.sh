#!/usr/bin/env bash
set -euo pipefail

TS="$(date +%Y%m%d_%H%M%S)"
ROOT="reports/livexml_tuning_${TS}"
LOG_FILE="${ROOT}/terminal_output_${TS}.log"
MANIFEST_FILE="${ROOT}/manifest_${TS}.json"
mkdir -p "${ROOT}"

print_header() {
  local title="$1"
  local hp="$2"
  local report="$3"
  local outdir="$4"
  echo "" | tee -a "${LOG_FILE}"
  echo "======================================================================" | tee -a "${LOG_FILE}"
  echo "RUN TITLE: ${title}" | tee -a "${LOG_FILE}"
  echo "TIMESTAMP: ${TS}" | tee -a "${LOG_FILE}"
  echo "HYPERPARAMETERS: ${hp}" | tee -a "${LOG_FILE}"
  echo "REPORT_PATH: ${report}" | tee -a "${LOG_FILE}"
  echo "OUTPUT_DIR: ${outdir}" | tee -a "${LOG_FILE}"
  echo "======================================================================" | tee -a "${LOG_FILE}"
}

run_one() {
  local title="$1"
  local hp="$2"
  local report="$3"
  local outdir="$4"
  shift 4
  print_header "${title}" "${hp}" "${report}" "${outdir}"
  python -m jellyfish.train "$@" --report-path "${report}" --output-dir "${outdir}" 2>&1 | tee -a "${LOG_FILE}"
}

echo "==============================================================" | tee -a "${LOG_FILE}"
echo "LIVE XML HYPERPARAMETER TUNING - ${TS}" | tee -a "${LOG_FILE}"
echo "==============================================================" | tee -a "${LOG_FILE}"

BASE_ARGS=(
  --models JellyfishNet
  --use-integrated-data
  --threshold-min 0.05
  --threshold-max 0.9
  --threshold-steps 81
  --threshold-min-precision 0.2
  --threshold-target-recall 0.95
)

run_one \
  "lx1_lb14_hd128_do020_lr7e4_bs24_pw1.4" \
  "lookback=14; hidden=128; dropout=0.2; lr=0.0007; bs=24; epochs=12; patience=8; sched_factor=0.2; sched_patience=1; pos_weight=1.4; include_live_xml=true" \
  "${ROOT}/lx1.json" \
  "models/livexml_${TS}/lx1" \
  "${BASE_ARGS[@]}" \
  --lookback-days 14 --hybrid-hidden-dim 128 --dropout-prob 0.2 --learning-rate 0.0007 --batch-size 24 --num-epochs 12 --patience 8 --scheduler-factor 0.2 --scheduler-patience 1 --positive-class-weight 1.4

run_one \
  "lx2_lb14_hd192_do015_lr6e4_bs16_pw1.6" \
  "lookback=14; hidden=192; dropout=0.15; lr=0.0006; bs=16; epochs=12; patience=8; sched_factor=0.2; sched_patience=1; pos_weight=1.6; include_live_xml=true" \
  "${ROOT}/lx2.json" \
  "models/livexml_${TS}/lx2" \
  "${BASE_ARGS[@]}" \
  --lookback-days 14 --hybrid-hidden-dim 192 --dropout-prob 0.15 --learning-rate 0.0006 --batch-size 16 --num-epochs 12 --patience 8 --scheduler-factor 0.2 --scheduler-patience 1 --positive-class-weight 1.6

run_one \
  "lx3_lb21_hd128_do015_lr5e4_bs16_pw1.4" \
  "lookback=21; hidden=128; dropout=0.15; lr=0.0005; bs=16; epochs=12; patience=8; sched_factor=0.3; sched_patience=1; pos_weight=1.4; include_live_xml=true" \
  "${ROOT}/lx3.json" \
  "models/livexml_${TS}/lx3" \
  "${BASE_ARGS[@]}" \
  --lookback-days 21 --hybrid-hidden-dim 128 --dropout-prob 0.15 --learning-rate 0.0005 --batch-size 16 --num-epochs 12 --patience 8 --scheduler-factor 0.3 --scheduler-patience 1 --positive-class-weight 1.4

run_one \
  "lx4_lb21_hd192_do010_lr5e4_bs16_pw1.6" \
  "lookback=21; hidden=192; dropout=0.10; lr=0.0005; bs=16; epochs=12; patience=8; sched_factor=0.3; sched_patience=1; pos_weight=1.6; include_live_xml=true" \
  "${ROOT}/lx4.json" \
  "models/livexml_${TS}/lx4" \
  "${BASE_ARGS[@]}" \
  --lookback-days 21 --hybrid-hidden-dim 192 --dropout-prob 0.10 --learning-rate 0.0005 --batch-size 16 --num-epochs 12 --patience 8 --scheduler-factor 0.3 --scheduler-patience 1 --positive-class-weight 1.6

run_one \
  "lx5_lb28_hd128_do020_lr4e4_bs16_pw1.4" \
  "lookback=28; hidden=128; dropout=0.20; lr=0.0004; bs=16; epochs=12; patience=8; sched_factor=0.3; sched_patience=1; pos_weight=1.4; include_live_xml=true" \
  "${ROOT}/lx5.json" \
  "models/livexml_${TS}/lx5" \
  "${BASE_ARGS[@]}" \
  --lookback-days 28 --hybrid-hidden-dim 128 --dropout-prob 0.20 --learning-rate 0.0004 --batch-size 16 --num-epochs 12 --patience 8 --scheduler-factor 0.3 --scheduler-patience 1 --positive-class-weight 1.4

run_one \
  "lx6_lb21_hd256_do015_lr4e4_bs12_pw1.8" \
  "lookback=21; hidden=256; dropout=0.15; lr=0.0004; bs=12; epochs=12; patience=8; sched_factor=0.2; sched_patience=1; pos_weight=1.8; include_live_xml=true" \
  "${ROOT}/lx6.json" \
  "models/livexml_${TS}/lx6" \
  "${BASE_ARGS[@]}" \
  --lookback-days 21 --hybrid-hidden-dim 256 --dropout-prob 0.15 --learning-rate 0.0004 --batch-size 12 --num-epochs 12 --patience 8 --scheduler-factor 0.2 --scheduler-patience 1 --positive-class-weight 1.8

python - <<'PY' "${ROOT}" "${MANIFEST_FILE}" "${TS}" | tee -a "${LOG_FILE}"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
ts = sys.argv[3]

run_files = sorted(root.glob("lx*.json"))
runs = []
for p in run_files:
    data = json.loads(p.read_text())
    m = data["results"]["JellyfishNet"]
    c = data["config"]
    runs.append({
        "run": p.stem,
        "report_path": str(p),
        "hyperparameters": c,
        "metrics": {
            "recall": m["recall"],
            "precision": m["precision"],
            "f1": m["f1"],
            "accuracy": m["accuracy"],
            "auc": m["auc"],
            "threshold": m["threshold"],
        },
        "curve_points": {
            "train_losses": len(m.get("train_losses", [])),
            "val_losses": len(m.get("val_losses", [])),
            "lr_history": len(m.get("lr_history", [])),
        }
    })

eligible = [r for r in runs if r["metrics"]["recall"] >= 0.95]
best = max(eligible, key=lambda r: (r["metrics"]["precision"], r["metrics"]["f1"], r["metrics"]["accuracy"])) if eligible else None

manifest = {
    "title": f"LIVE XML HYPERPARAMETER TUNING - {ts}",
    "timestamp": ts,
    "objective": "maximize precision/f1 while keeping recall >= 0.95",
    "runs": runs,
    "best_under_recall_constraint": best,
}
manifest_path.write_text(json.dumps(manifest, indent=2))
print(f"Wrote {manifest_path}")
PY

echo "" | tee -a "${LOG_FILE}"
echo "DONE. LOG: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "DONE. MANIFEST: ${MANIFEST_FILE}" | tee -a "${LOG_FILE}"
