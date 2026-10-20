#!/usr/bin/env bash
# PC 侧入口：统一环境检查后再跑具体脚本。
# 不做虚拟环境自动创建——依赖只有 numpy，不值得引入复杂度。
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python3}
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "找不到 $PY，用 PY=/path/to/python 指定" >&2
  exit 1
fi

case "${1:-help}" in
  env)      "$PY" python/check_env.py ;;
  eval)     "$PY" python/build_eval_set.py ;;
  parity)   "$PY" python/mel_parity.py ;;
  fsm)      "$PY" python/fsm.py ;;
  probe)    "$PY" python/probe_operators.py ;;
  infer)    shift; "$PY" python/board_infer.py "$@" ;;
  power)    shift; "$PY" python/board_power_budget.py "$@" ;;
  all)
    "$PY" python/check_env.py
    "$PY" python/build_eval_set.py
    "$PY" python/mel_parity.py
    "$PY" python/fsm.py
    "$PY" python/probe_operators.py
    ;;
  *)
    cat <<'EOF'
用法: ./scripts/dev.sh <command>

  env      环境自检
  eval     生成合成评测集
  parity   前端一致性归因（核心）
  fsm      状态机自检
  probe    算子风险清单
  infer    板端/PC 流式推理打点   (参数透传)
  power    功耗账本               (参数透传)
  all      依次跑 env/eval/parity/fsm/probe
EOF
    ;;
esac
