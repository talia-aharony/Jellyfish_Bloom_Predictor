#!/usr/bin/env bash
set -euo pipefail

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="reports/next_steps_${TS}"
LOG_FILE="${LOG_DIR}/terminal_output_${TS}.log"
MANIFEST_FILE="${LOG_DIR}/manifest_${TS}.json"
mkdir -p "${LOG_DIR}"

echo "==============================================================" | tee -a "${LOG_FILE}"
echo "NEXT-STEPS DIAGNOSTIC LOOP - ${TS}" | tee -a "${LOG_FILE}"
echo "==============================================================" | tee -a "${LOG_FILE}"

run_train () {
  local title="$1"; shift
  local report_path="$1"; shift
  local output_dir="$1"; shift
  local hp_header="$1"; shift

  echo "" | tee -a "${LOG_FILE}"
  echo "--------------------------------------------------------------" | tee -a "${LOG_FILE}"
  echo "RUN TITLE: ${title}" | tee -a "${LOG_FILE}"
  echo "RUN TIMESTAMP: ${TS}" | tee -a "${LOG_FILE}"
  echo "HYPERPARAMETERS: ${hp_header}" | tee -a "${LOG_FILE}"
  echo "REPORT_PATH: ${report_path}" | tee -a "${LOG_FILE}"
  echo "OUTPUT_DIR: ${output_dir}" | tee -a "${LOG_FILE}"
  echo "--------------------------------------------------------------" | tee -a "${LOG_FILE}"

  python -m jellyfish.train "$@" \
    --report-path "${report_path}" \
    --output-dir "${output_dir}" 2>&1 | tee -a "${LOG_FILE}"
}

run_train \
  "Step1_LongHorizon_SchedulerFirst_${TS}" \
  "${LOG_DIR}/step1_longhorizon_scheduler.json" \
  "models/next_steps_${TS}/step1" \
  "model=JellyfishNet; integrated=true; lookback=14; epochs=20; patience=10; lr=0.001; bs=32; dropout=0.3; hidden=64; scheduler_factor=0.2; scheduler_patience=1; threshold_min=0.05; threshold_max=0.9; threshold_steps=81; threshold_min_precision=0.2; threshold_target_recall=0.95; positive_class_weight=1.0; include_live_xml=false" \
  --models JellyfishNet \
  --use-integrated-data \
  --disable-live-xml \
  --lookback-days 14 \
  --num-epochs 20 \
  --patience 10 \
  --learning-rate 0.001 \
  --batch-size 32 \
  --dropout-prob 0.3 \
  --hybrid-hidden-dim 64 \
  --scheduler-factor 0.2 \
  --scheduler-patience 1 \
  --threshold-min 0.05 \
  --threshold-min-precision 0.2 \
  --threshold-target-recall 0.95 \
  --positive-class-weight 1.0

run_train \
  "Step2_RecallAwareLoss_${TS}" \
  "${LOG_DIR}/step2_recall_weighted.json" \
  "models/next_steps_${TS}/step2" \
  "model=JellyfishNet; integrated=true; lookback=14; epochs=20; patience=10; lr=0.001; bs=32; dropout=0.3; hidden=64; scheduler_factor=0.2; scheduler_patience=1; threshold_min=0.05; threshold_max=0.9; threshold_steps=81; threshold_min_precision=0.2; threshold_target_recall=0.95; positive_class_weight=1.6; include_live_xml=false" \
  --models JellyfishNet \
  --use-integrated-data \
  --disable-live-xml \
  --lookback-days 14 \
  --num-epochs 20 \
  --patience 10 \
  --learning-rate 0.001 \
  --batch-size 32 \
  --dropout-prob 0.3 \
  --hybrid-hidden-dim 64 \
  --scheduler-factor 0.2 \
  --scheduler-patience 1 \
  --threshold-min 0.05 \
  --threshold-min-precision 0.2 \
  --threshold-target-recall 0.95 \
  --positive-class-weight 1.6

run_train \
  "Step3_Lookback21_StructuralRecallAware_${TS}" \
  "${LOG_DIR}/step3_lookback21_structural_weighted.json" \
  "models/next_steps_${TS}/step3" \
  "model=JellyfishNet; integrated=true; lookback=21; epochs=20; patience=10; lr=0.0007; bs=24; dropout=0.2; hidden=128; scheduler_factor=0.2; scheduler_patience=1; threshold_min=0.05; threshold_max=0.9; threshold_steps=81; threshold_min_precision=0.2; threshold_target_recall=0.95; positive_class_weight=1.6; include_live_xml=false" \
  --models JellyfishNet \
  --use-integrated-data \
  --disable-live-xml \
  --lookback-days 21 \
  --num-epochs 20 \
  --patience 10 \
  --learning-rate 0.0007 \
  --batch-size 24 \
  --dropout-prob 0.2 \
  --hybrid-hidden-dim 128 \
  --scheduler-factor 0.2 \
  --scheduler-patience 1 \
  --threshold-min 0.05 \
  --threshold-min-precision 0.2 \
  --threshold-target-recall 0.95 \
  --positive-class-weight 1.6

python - <<'PY' "${LOG_DIR}" "${MANIFEST_FILE}" "${TS}" | tee -a "${LOG_FILE}"
import json
import sys
from pathlib import Path

log_dir = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
ts = sys.argv[3]

run_files = [
    ("step1_longhorizon_scheduler", log_dir / "step1_longhorizon_scheduler.json"),
    ("step2_recall_weighted", log_dir / "step2_recall_weighted.json"),
    ("step3_lookback21_structural_weighted", log_dir / "step3_lookback21_structural_weighted.json"),
]

summary = {
    "title": f"NEXT-STEPS DIAGNOSTIC LOOP - {ts}",
    "timestamp": ts,
    "objective": "maximize precision/f1 while keeping recall >= 0.95",
    "runs": []
}

for label, path in run_files:
    data = json.loads(path.read_text())
    m = data["results"]["JellyfishNet"]
    c = data["config"]
    summary["runs"].append({
        "label": label,
        "report_path": str(path),
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

eligible = [r for r in summary["runs"] if r["metrics"]["recall"] >= 0.95]
summary["best_under_recall_constraint"] = (
    max(eligible, key=lambda r: (r["metrics"]["precision"], r["metrics"]["f1"], r["metrics"]["accuracy"])) if eligible else None
)

manifest_path.write_text(json.dumps(summary, indent=2))
print(f"Wrote {manifest_path}")
PY

echo "" | tee -a "${LOG_FILE}"
echo "DONE. TERMINAL OUTPUT LOG: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "DONE. MANIFEST: ${MANIFEST_FILE}" | tee -a "${LOG_FILE}"
