#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel)}"
DEFAULT_TASK_ID="${DEFAULT_TASK_ID:-human_3}"
MAX_SMOKE_TASKS="${MAX_SMOKE_TASKS:-3}"

candidates=$(grep -E '^(data|metric)/[^/]+/' || true)
candidates=$(printf '%s\n' "$candidates" | sed -E 's|^(data\|metric)/([^/]+)/.*|\2|' | grep -v '^$' | sort -u || true)

selected=()
for task_id in $candidates; do
  metric_file="$REPO_ROOT/metric/$task_id/metric.yaml"
  if [ ! -f "$metric_file" ] || [ ! -d "$REPO_ROOT/data/$task_id/gt" ]; then
    echo "Skipping $task_id: no metric/$task_id/metric.yaml or data/$task_id/gt" >&2
    continue
  fi
  if grep -qE 'vlm_utils|llm_utils' "$metric_file"; then
    echo "Skipping $task_id: its metric needs an LLM or VLM judge" >&2
    continue
  fi
  selected+=("$task_id")
done

if [ "${#selected[@]}" -eq 0 ]; then
  echo "No evaluable task changed, falling back to $DEFAULT_TASK_ID" >&2
  selected=("$DEFAULT_TASK_ID")
fi

if [ "${#selected[@]}" -gt "$MAX_SMOKE_TASKS" ]; then
  echo "Changed tasks: ${#selected[@]}, the smoke run is truncated to the first $MAX_SMOKE_TASKS: ${selected[*]:0:$MAX_SMOKE_TASKS}" >&2
  selected=("${selected[@]:0:$MAX_SMOKE_TASKS}")
fi

echo "${selected[*]}"
