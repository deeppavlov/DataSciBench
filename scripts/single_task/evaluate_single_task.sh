#!/bin/bash

# Specify TASK_ID here (same as in run_single_task.sh)
TASK_ID="human_3"
# Model name for evaluation
MODEL_NAME="gemma-3-27b-it"

LOG_DIR="single_task_run"
LOG_FILE="$LOG_DIR/eval_${TASK_ID}.log"

mkdir -p "$LOG_DIR"
> "$LOG_FILE"

echo "==========================================================" | tee -a "$LOG_FILE"
echo "Evaluating single task: $TASK_ID for model $MODEL_NAME" | tee -a "$LOG_FILE"
echo "Logs will be saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "==========================================================" | tee -a "$LOG_FILE"

# Run evaluation
python -m experiments.evaluate --task_id "$TASK_ID" --model_id "$MODEL_NAME" 2>&1 | tee -a "$LOG_FILE"

echo "==========================================================" | tee -a "$LOG_FILE"
echo "Evaluation complete." | tee -a "$LOG_FILE"
