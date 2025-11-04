#!/bin/bash
# Usage: ./fetch_outputs.sh <PORT> <IP>
# Downloads the outputs/2025-10-30_18-23-32 directory from a remote machine
# to the local ./outputs/ directory.

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <SSH_PORT> <REMOTE_IP>"
  exit 1
fi

PORT=$1
IP=$2
REMOTE_DIR="/workspace/practical-work/outputs/*"
LOCAL_DIR="./outputs/"
IDENTITY_FILE="$HOME/.ssh/id_ed25519_vast_ai"

mkdir -p "$LOCAL_DIR"

rsync -avz \
  -e "ssh -p $PORT -i $IDENTITY_FILE" \
  root@$IP:"$REMOTE_DIR" "$LOCAL_DIR"