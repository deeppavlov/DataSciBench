#!/bin/bash
. "$(dirname "$0")/../_python.sh" 2>/dev/null || PY="${PYTHON:-python}"

CONFIG_FILE="gemma3.yaml"
LOG_DIR="single_task_run"

mkdir -p "$LOG_DIR"

TASKS=(
    "dl_9"
)

TOTAL=${#TASKS[@]}

echo "=========================================================="
echo "Starting batch run of $TOTAL tasks"
echo "=========================================================="

for i in "${!TASKS[@]}"; do
    TASK_ID="${TASKS[$i]}"
    NUM=$((i + 1))
    DATA_TYPE=$(echo "$TASK_ID" | sed 's/_[0-9]*$//')
    LOG_FILE="$LOG_DIR/run_${TASK_ID}.log"

    > "$LOG_FILE"

    echo "==========================================================" | tee -a "$LOG_FILE"
    echo "[$NUM/$TOTAL] Running task: $TASK_ID (data_type: $DATA_TYPE)" | tee -a "$LOG_FILE"
    echo "Logs will be saved to: $LOG_FILE" | tee -a "$LOG_FILE"
    echo "==========================================================" | tee -a "$LOG_FILE"

    "$PY" -m experiments.run_examples --task_id "$TASK_ID" --max_runs 10 --config "$CONFIG_FILE" --data_type "$DATA_TYPE" 2>&1 | tee -a "$LOG_FILE"

    echo "==========================================================" | tee -a "$LOG_FILE"
    echo "[$NUM/$TOTAL] Task $TASK_ID finished." | tee -a "$LOG_FILE"
    echo "==========================================================" | tee -a "$LOG_FILE"
    echo ""
done

echo "=========================================================="
echo "All $TOTAL tasks finished."
echo "=========================================================="
