#!/bin/bash

# Specify TASK_ID here (e.g., from data folder: dl_0, human_12, etc.)
TASK_ID="code_gen_001"
# Config for the run
CONFIG_FILE="gemma3.yaml"

LOG_DIR="single_task_run"
LOG_FILE="$LOG_DIR/run_${TASK_ID}.log"

mkdir -p "$LOG_DIR"
> "$LOG_FILE"

echo "==========================================================" | tee -a "$LOG_FILE"
echo "Running single task: $TASK_ID" | tee -a "$LOG_FILE"
echo "Logs will be saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "==========================================================" | tee -a "$LOG_FILE"

# Run generation for a single task. We set data_source_type and data_type to empty to avoid filtering.
python -m experiments.run_examples --task_id "$TASK_ID" --data_source_type "" --data_type "" --max_runs 1 --config "$CONFIG_FILE" 2>&1 | tee -a "$LOG_FILE"

echo "==========================================================" | tee -a "$LOG_FILE"
echo "Run complete." | tee -a "$LOG_FILE"
