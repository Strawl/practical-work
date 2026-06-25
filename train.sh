#!/usr/bin/env bash

unset LD_LIBRARY_PATH
set -euo pipefail

INPUT_DIR="./train_configs"
OUTPUT_BASE="./terminal_output"

mkdir -p "$OUTPUT_BASE"

for config in "$INPUT_DIR"/*; do
  [ -f "$config" ] || continue

  name="$(basename "$config")"
  run_dir="$OUTPUT_BASE/${name%.*}"

  mkdir -p "$run_dir"

  # Run training, capturing the save directory from output
  SAVE_DIR=$(uv run topopt train --config "$config" 2>&1 | tee "$run_dir/output.log" | grep "Saving data to:" | sed 's/Saving data to: //')

  if [ -n "$SAVE_DIR" ]; then
    echo "Training complete. Evaluating models in: $SAVE_DIR" | tee -a "$run_dir/output.log"
    uv run topopt evaluate --save-dir "$SAVE_DIR" --scale 15 >> "$run_dir/output.log" 2>&1
  else
    echo "ERROR: Could not determine save directory for $name" | tee -a "$run_dir/output.log"
  fi
done