#!/usr/bin/env bash

set -euo pipefail

OUTPUT_BASE="./outputs"
EVAL_SCALE=15

for run_dir in "$OUTPUT_BASE"/*; do
  # Only process directories
  [ -d "$run_dir" ] || continue

  echo "Evaluating run: $run_dir"

  uv run topopt evaluate \
    --scale "$EVAL_SCALE" \
    --save-dir "$run_dir" \
    > "$run_dir/evaluate.log" 2>&1
done