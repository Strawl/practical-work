#!/bin/bash
# Usage: ./sync_practical.sh <PORT> <IP>
# Syncs the current directory to /root/practical-work/ on a remote machine,
# excluding all files listed in .gitignore

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <SSH_PORT> <REMOTE_IP>"
  exit 1
fi

PORT=$1
IP=$2
DEST_DIR="/workspace/practical-work/"
IDENTITY_FILE="$HOME/.ssh/id_ed25519_vast_ai"

EXCLUDE_FILE=$(mktemp)
{
  git status --ignored --short | awk '/^!! /{print substr($0,4)}'
} > "$EXCLUDE_FILE"

rsync -avz --delete \
  -e "ssh -p $PORT -i $IDENTITY_FILE" \
  --exclude-from="$EXCLUDE_FILE" \
  ./ root@$IP:$DEST_DIR

# Clean up temp file
rm -f "$EXCLUDE_FILE"