#!/usr/bin/env bash

set -euo pipefail

INPUT_DIR="./train_configs"
OUTPUT_BASE="./outputs/terminal_output"

mkdir -p "$OUTPUT_BASE"

for config in "$INPUT_DIR"/*; do
  [ -f "$config" ] || continue

  name="$(basename "$config")"
  run_dir="$OUTPUT_BASE/${name%.*}"

  mkdir -p "$run_dir"

  uv run topopt train --config "$config" \
    >"$run_dir/output.log" 2>&1
done