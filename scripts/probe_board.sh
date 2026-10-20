#!/usr/bin/env bash
# 板端摸底：一次性把"这块板能不能做这个项目"的答案打出来。
# 拿板第一件事就跑这个，别急着装模型。
set -uo pipefail

echo "=== A311Y3 板端摸底 ==="
echo "kernel : $(uname -a)"
echo "model  : $(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo unknown)"
echo

echo "--- NPU 运行时 ---"
ls -l /usr/lib/libnnsdk*.so* /usr/lib/libadla*.so* /vendor/lib/libnnsdk* 2>/dev/null || echo "  未找到 nnsdk/adla 库"
for p in /sys/kernel/debug/adla/version /sys/class/misc/adla/version; do
  [ -r "$p" ] && echo "  driver: $(cat "$p")"
done
echo

echo "--- ALSA 采集 ---"
if command -v arecord >/dev/null 2>&1; then
  arecord -l 2>/dev/null | sed 's/^/  /'
else
  echo "  未安装 alsa-utils（apt install alsa-utils）"
fi
echo

echo "--- cpuidle（功耗账本前提）---"
found=0
for c in /sys/devices/system/cpu/cpu[0-9]*/cpuidle; do
  [ -d "$c" ] || continue
  found=1
  for s in "$c"/state*; do
    [ -r "$s/name" ] || continue
    printf "  %-12s %-12s usage=%s\n" "$(basename "$(dirname "$c")")" \
      "$(cat "$s/name")" "$(cat "$s/usage" 2>/dev/null || echo -)"
  done
  break
done
[ "$found" -eq 0 ] && echo "  无 cpuidle 节点：改用 /proc/stat 空闲计数做替代口径"
echo

echo "--- cpufreq ---"
for f in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor; do
  [ -r "$f" ] || continue
  echo "  $(dirname "$(dirname "$(dirname "$f")")" | xargs basename): $(cat "$f")"
done | head -8
echo

echo "--- 温度 / 内存 ---"
for t in /sys/class/thermal/thermal_zone*/temp; do
  [ -r "$t" ] && echo "  $(basename "$(dirname "$t")"): $(( $(cat "$t") / 1000 )) °C"
done | head -4
free -m | sed 's/^/  /'
echo

echo "--- 音频时钟（判 xrun 风险）---"
cat /proc/asound/card*/pcm*p/sub0/hw_params 2>/dev/null | head -20 || echo "  无活动 PCM"
echo

echo "=== 下一步 ==="
echo "1. 有 nnsdk 库 → cmake -B build -DUSE_ADLA=ON -DNNSDK_ROOT=... 构建 cpp/"
echo "2. 无 nnsdk   → 先用 python/board_infer.py --wav 打通链路与打点"
echo "3. 跑 python/probe_operators.py --try-convert 拿算子实测真值"
