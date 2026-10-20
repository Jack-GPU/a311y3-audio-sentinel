"""环境自检：PC 侧与板端共用。

先把"能不能跑"确认掉，再谈数字。板端尤其要看三件事：
ALSA 采集是否可用、cpuidle 是否可见（功耗账本的前提）、NPU runtime 版本。

用法::

    python check_env.py            # PC 或板端
    python check_env.py --board    # 板端额外检查
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def line(ok: bool | None, name: str, detail: str = "") -> None:
    mark = "OK  " if ok else ("WARN" if ok is None else "FAIL")
    print(f"  [{mark}] {name:<26}{detail}")


def check_python() -> None:
    print("\n-- 运行时 --")
    line(sys.version_info >= (3, 9), "python", platform.python_version())
    try:
        import numpy
        line(True, "numpy", numpy.__version__)
    except ImportError:
        line(False, "numpy", "pip install numpy")


def check_toolkit() -> None:
    print("\n-- 转换工具链（PC 侧）--")
    try:
        import onnx
        line(True, "onnx", onnx.__version__)
    except ImportError:
        line(None, "onnx", "导出模型需要：pip install onnx")
    try:
        import amlnn
        ver = getattr(amlnn, "__version__", "unknown")
        line(True, "amlnn-toolkit", ver)
    except ImportError:
        line(None, "amlnn-toolkit", "转换 .adla 需要 AMLNN Toolkit")


def check_board() -> None:
    print("\n-- 板端 --")

    # NPU runtime
    libs = []
    for pat in ("/usr/lib/libnnsdk*.so*", "/usr/lib/libadla*.so*", "/vendor/lib/libnnsdk*"):
        libs += [str(p) for p in Path("/").glob(pat.lstrip("/"))]
    line(bool(libs), "NPU runtime libs", ", ".join(Path(p).name for p in libs[:3]) or "未找到")

    for p in ("/sys/kernel/debug/adla/version", "/sys/class/misc/adla/version"):
        if Path(p).exists():
            try:
                line(True, "ADLA driver", Path(p).read_text().strip())
            except OSError:
                line(None, "ADLA driver", f"{p} 不可读（需 root）")
            break
    else:
        line(None, "ADLA driver", "未发现版本节点，手工确认 >= 2.0.2")

    # ALSA
    arecord = shutil.which("arecord")
    if arecord:
        r = subprocess.run([arecord, "-l"], capture_output=True, text=True)
        cap = [l for l in r.stdout.splitlines() if "card" in l]
        line(bool(cap), "ALSA capture", f"{len(cap)} 个设备")
        for c in cap[:4]:
            print(f"         {c.strip()}")
    else:
        line(None, "ALSA capture", "未安装 alsa-utils")

    # cpuidle（功耗账本前提）
    cpu_root = Path("/sys/devices/system/cpu")
    idle = list(cpu_root.glob("cpu[0-9]*/cpuidle/state*"))
    line(bool(idle), "cpuidle nodes", f"{len(idle)} 个 state（功耗账本依赖）")
    if not idle:
        print("         若为空，考察 /proc/stat 空闲计数作为替代口径")

    # cpufreq
    freq = list(cpu_root.glob("cpu[0-9]*/cpufreq"))
    line(bool(freq), "cpufreq", f"{len(freq)} 个核可调频")

    # 温度
    for tp in ("/sys/class/thermal/thermal_zone0/temp",):
        if Path(tp).exists():
            try:
                line(True, "thermal zone0", f"{int(Path(tp).read_text()) / 1000:.1f} °C")
            except (OSError, ValueError):
                line(None, "thermal zone0", "读取失败")
            break

    # 关键 sysfs：是否能看到我们关心的域
    line(None, "HiFi5 / always-on 核",
         "需查 SDK 文档与 dmesg；不可编程时按 方案 v2 降级路径走 A55")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", action="store_true")
    args = ap.parse_args()

    print("=== A311Y3 音频感知引擎 · 环境自检 ===")
    print(f"host: {platform.platform()}  cwd: {os.getcwd()}")
    check_python()
    check_toolkit()
    if args.board:
        check_board()

    print("\n版本纪律：toolkit / ADLA driver / NNSDK2 / target_platform(007) 必须成对。")
    print("指纹写入模型同目录的 *.fingerprint.json，交付时一并附上。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
