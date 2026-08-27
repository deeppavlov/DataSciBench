#!/usr/bin/env bash
. "$(dirname "$0")/_python.sh" 2>/dev/null || PY="${PYTHON:-python}"
cd "$(dirname "$0")/.." || exit 1
task=$1
model=${2:-claude}
mkdir -p logs/enchancing
for d in data/"$task"/"$model"_*; do
  [ -d "$d" ] || continue
  if [ ! -s "$d/logs.txt" ]; then echo "run by orchestrator subagent" > "$d/logs.txt"; fi
done
exec 9>logs/enchancing/.eval.lock
flock -w 7200 9 || { echo "не дождался блокировки"; exit 1; }
"$PY" -m experiments.evaluate --task_id "$task" --model_id "$model" > "logs/enchancing/${task}_${model}_eval.log" 2>&1
rc=$?
echo "rc=$rc"
if [ "$rc" -ne 0 ]; then
  rm -f "logs/enchancing/${task}_${model}_results.csv"
  echo "оценка упала, результатов нет, конец лога:"
  tail -n 20 "logs/enchancing/${task}_${model}_eval.log"
  exit "$rc"
fi
cp "evaluation_results/results/${model}_results.csv" "logs/enchancing/${task}_${model}_results.csv"
grep ",${task}," "logs/enchancing/${task}_${model}_results.csv" || echo "нет строк для ${task}"
