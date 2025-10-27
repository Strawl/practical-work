#!/bin/bash
# Usage: ./sync_practical.sh <PORT> <IP>
# Syncs current directory to /root/practical-work/ on a remote machine via rsync.

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <SSH_PORT> <REMOTE_IP>"
  exit 1
fi

PORT=$1
IP=$2
DEST_DIR="/workspace/practical-work"
IDENTITY_FILE="$HOME/.ssh/id_ed25519_vast_ai"

# Rsync command (dry-run for safety)
rsync -avz --delete \
  -e "ssh -p $PORT -i $IDENTITY_FILE" \
  --exclude '.venv/' \
  --exclude '__pychache__' \
  --exclude '.git/' \
  ./ root@$IP:$DEST_DIR