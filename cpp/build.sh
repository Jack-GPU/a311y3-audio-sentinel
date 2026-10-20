#!/usr/bin/env bash
# 板端构建。依赖：cmake >= 3.13、libasound2-dev、gcc。
set -euo pipefail
cd "$(dirname "$0")"

BUILD=${BUILD:-build}

if ! command -v cmake >/dev/null 2>&1; then
  echo "缺 cmake：apt install cmake build-essential libasound2-dev" >&2
  exit 1
fi

cmake -B "$BUILD" -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build "$BUILD" -j"$(nproc 2>/dev/null || echo 2)"

echo
echo "产物："
ls -lh "$BUILD"/audio_sentinel "$BUILD"/mel_dump 2>/dev/null || true
echo
echo "跑："
echo "  $BUILD/audio_sentinel --device hw:0,0 --secs 10"
echo "  $BUILD/mel_dump ../data/wav/glass_break_clean.wav > c_mel.txt"
