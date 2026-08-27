#!/bin/bash
. "$(dirname "$0")/_python.sh" 2>/dev/null || PY="${PYTHON:-python}"
data_types=("human" "csv_excel" "dl")

for data_type in "${data_types[@]}"; do
    echo "Running data type: $data_type"
    "$PY" -m experiments.run_examples --data_type $data_type --max_runs 3 --config gemma3.yaml
done

echo "All tests completed."
