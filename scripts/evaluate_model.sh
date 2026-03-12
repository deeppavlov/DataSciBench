#!/bin/bash

# You can pass a model name as the first argument. Defaults to "gemma-3-27b-it"
MODEL_NAME=${1:-"Qwen3-30B-A3B"}

echo "=========================================================="
echo "Starting Evaluation Pipeline for model: $MODEL_NAME"
echo "=========================================================="

echo "[Prep] Clearing old VLM logs..."
rm -f evaluation_results/vlm_run_log.txt

echo "[Step 1] Parsing outputs and running VLM metric scoring..."
python -m experiments.evaluate --task_id all --model_id "$MODEL_NAME"
if [ $? -ne 0 ]; then
    echo "Error during Step 1 evaluation. Exiting."
    exit 1
fi

echo ""
echo "[Step 2] Calculating final CSV metrics..."
python -m evaluation_results.calculate_final_metric
if [ $? -ne 0 ]; then
    echo "Error during Step 2 final metric calculation. Exiting."
    exit 1
fi

echo ""
echo "=========================================================="
echo "Evaluation Pipeline Complete."
echo "Final results have been updated."
echo "VLM Logs are now unified and located at:"
echo "evaluation_results/vlm_run_log.txt"
echo "=========================================================="
