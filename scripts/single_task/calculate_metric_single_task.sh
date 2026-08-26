#!/bin/bash
. "$(dirname "$0")/../_python.sh" 2>/dev/null || PY="${PYTHON:-python}"

LOG_DIR="single_task_run"
LOG_FILE="$LOG_DIR/calc_metric.log"

mkdir -p "$LOG_DIR"
> "$LOG_FILE"

echo "==========================================================" | tee -a "$LOG_FILE"
echo "Calculating final metrics (calculate_final_metric.py)" | tee -a "$LOG_FILE"
echo "Logs will be saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "==========================================================" | tee -a "$LOG_FILE"

# Calculate final metric
"$PY" -m evaluation_results.calculate_final_metric 2>&1 | tee -a "$LOG_FILE"

echo "==========================================================" | tee -a "$LOG_FILE"
echo "Metric calculation complete." | tee -a "$LOG_FILE"
