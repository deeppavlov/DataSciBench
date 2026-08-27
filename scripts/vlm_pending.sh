#!/usr/bin/env bash
cd "$(dirname "$0")/.." || exit 1
for req in evaluation_results/vlm_bridge/*.req.json; do
  [ -e "$req" ] || continue
  key=$(basename "$req" .req.json)
  [ -e "evaluation_results/vlm_bridge/$key.score" ] && continue
  echo "$req"
done
