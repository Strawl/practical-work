#!/bin/bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: sync_studentwork.sh <target|studentNN|NN> [remote_subdir]

Sync the current directory to /system/user/studentwork/<remote_subdir> on a remote
server using rsync and the existing SSH configuration.

Examples:
  sync_studentwork.sh student03
  sync_studentwork.sh 3
  sync_studentwork.sh student07 experiment-a
  sync_studentwork.sh gpu-box shared/project
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  usage
  exit 1
fi

normalize_target() {
  local raw="$1"

  if [[ "$raw" =~ ^[0-9]+$ ]]; then
    printf 'student%02d\n' "$raw"
    return
  fi

  if [[ "$raw" =~ ^student([0-9]+)$ ]]; then
    printf 'student%02d\n' "${BASH_REMATCH[1]}"
    return
  fi

  printf '%s\n' "$raw"
}

TARGET=$(normalize_target "$1")
LOCAL_DIR=$(pwd)
LOCAL_BASENAME=$(basename "$LOCAL_DIR")
REMOTE_SUBDIR=${2:-$LOCAL_BASENAME}
REMOTE_ROOT="/system/user/studentwork/gyarmati"
REMOTE_DIR="$REMOTE_ROOT/$REMOTE_SUBDIR"

EXCLUDE_FILE=$(mktemp)
cleanup() {
  rm -f "$EXCLUDE_FILE"
}
trap cleanup EXIT

{
  printf '.git/\n'
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git status --ignored --short | awk '/^!! /{print substr($0,4)}'
  fi
} > "$EXCLUDE_FILE"

echo "Syncing $LOCAL_DIR to $TARGET:$REMOTE_DIR"

ssh "$TARGET" "mkdir -p '$REMOTE_DIR'"

rsync -avz --delete \
  --exclude-from="$EXCLUDE_FILE" \
  ./ "$TARGET:$REMOTE_DIR/"
