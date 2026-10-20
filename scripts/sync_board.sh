#!/usr/bin/env bash
# 同步到板子。只传代码与配置，模型与大件单独传（体积差一个数量级）。
set -euo pipefail
cd "$(dirname "$0")/.."

BOARD=${BOARD:-root@192.168.1.100}
DEST=${DEST:-/root/a311y3-audio}

echo "→ $BOARD:$DEST"

rsync -az --delete \
  --exclude '.git' --exclude '__pycache__' --exclude 'build' \
  --exclude 'data/wav/*.wav' --exclude 'models/*.adla' --exclude 'models/*.onnx' \
  python configs scripts docs "$BOARD:$DEST/"

echo "→ 代码已同步。模型请单独传："
echo "  scp models/*.adla $BOARD:$DEST/models/"
echo
echo "板端首次运行："
echo "  ssh $BOARD"
echo "  cd $DEST && bash scripts/probe_board.sh"
echo "  cd cpp && cmake -B build && cmake --build build -j"
