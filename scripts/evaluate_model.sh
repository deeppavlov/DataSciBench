#!/bin/bash
. "$(dirname "$0")/_python.sh" 2>/dev/null || PY="${PYTHON:-python}"
cd "$(dirname "$0")/.." || exit 1

# You can pass a model name as the first argument. Defaults to "gemma-3-27b-it"
MODEL_NAME=${1:-"Qwen3-30B-A3B"}

echo "=========================================================="
echo "Starting Evaluation Pipeline for model: $MODEL_NAME"
echo "=========================================================="

echo "[Step 1] Parsing outputs and running VLM metric scoring..."
"$PY" -m experiments.evaluate --task_id all --model_id "$MODEL_NAME"
if [ $? -ne 0 ]; then
    echo "Error during Step 1 evaluation. Exiting."
    exit 1
fi

# echo ""
# echo "[Step 2] Calculating final CSV metrics..."
# "$PY" -m evaluation_results.calculate_final_metric
# if [ $? -ne 0 ]; then
#     echo "Error during Step 2 final metric calculation. Exiting."
#     exit 1
# fi

# echo ""
# echo "=========================================================="
# echo "Evaluation Pipeline Complete."
# echo "Final results have been updated."
# echo "=========================================================="
