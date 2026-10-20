"""ONNX → ADLA3 转换封装。

把版本纪律写进代码，而不是写在 README 里靠人记：
toolkit / driver / runtime / target_platform 四项必须成对，
和 RKNN 项目踩过的坑是同一个坑，只是换了一条工具链。

用法::

    python convert_adla.py --onnx models/bcresnet_s.onnx \
        --calib data/calib/wav_list.txt --out models/bcresnet_s_w8a8.adla
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

TARGET_PLATFORM = "007"  # A311Y3 / ADLA3
TARGET_NAME = "A311Y3"
MIN_DRIVER = "2.0.2"
MIN_NNSDK = "3.0.0"
MIN_NNSDK2 = "1.0.0"


def fingerprint() -> dict:
    """记录本次转换的工具链指纹，产物旁边必须有一份。"""
    import platform

    info = {"target_platform": TARGET_PLATFORM, "target_name": TARGET_NAME,
            "python": platform.python_version(), "host": platform.platform(),
            "amlnn": None}
    try:
        import amlnn
        info["amlnn"] = getattr(amlnn, "__version__", "unknown")
    except ImportError:
        pass
    return info


def check_board_runtime(host: str | None) -> dict:
    """板端版本体检（可选，需 ssh 可达）。"""
    if not host:
        return {"checked": False}
    cmds = {
        "driver": "cat /sys/kernel/debug/adla/version 2>/dev/null || echo unknown",
        "nnsdk": "ls /usr/lib/libnnsdk.so* 2>/dev/null | head -1",
        "nnsdk2": "ls /usr/lib/libnnsdk2.so* 2>/dev/null | head -1",
    }
    out = {"checked": True, "host": host}
    for k, c in cmds.items():
        try:
            r = subprocess.run(["ssh", host, c], capture_output=True, text=True, timeout=10)
            out[k] = (r.stdout or r.stderr).strip()
        except Exception as exc:  # noqa: BLE001
            out[k] = f"error: {exc}"
    return out


def convert(onnx: Path, calib: Path | None, out: Path, dtype: str, mean, std) -> int:
    try:
        import amlnn
    except ImportError:
        print("[fail] 未安装 amlnn-toolkit。请在 PC 侧 toolkit 环境运行本脚本。")
        return 2

    if not onnx.exists():
        print(f"[fail] ONNX 不存在：{onnx}")
        return 2

    print(f"[1/4] load_onnx {onnx}")
    model = amlnn.load_onnx(str(onnx))

    kw = {
        "normalization_mean": [[mean]],
        "normalization_std": [[std]],
        "quantized_dtype": dtype,
        "target_platform": f"PRODUCT_PID0XA{TARGET_PLATFORM}",
    }
    if calib:
        kw["dataset"] = str(calib)
        print(f"[2/4] config dtype={dtype} target={TARGET_PLATFORM} calib={calib}")
    else:
        print(f"[2/4] config dtype={dtype} target={TARGET_PLATFORM} (no calib!)")
    amlnn.config(model, **kw)

    print("[3/4] compile")
    amlnn.compile(model)

    print(f"[4/4] export_adla → {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    amlnn.export_adla(model, str(out))

    side = out.with_suffix(".fingerprint.json")
    side.write_text(json.dumps(fingerprint(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] 产物 {out.stat().st_size / 1e6:.2f} MB，指纹 {side}")
    print("\n⚠️ 板端必须满足：ADLA driver >= "
          f"{MIN_DRIVER} / NNSDK >= {MIN_NNSDK} / NNSDK2 >= {MIN_NNSDK2}")
    print("   版本不成对时典型表现是加载成功但输出全 0 或形状错乱。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--calib", type=Path, default=None, help="校准列表，每行一个 wav 路径")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dtype", default="w8a8", choices=["w8a8", "w8a16", "w16a16"])
    ap.add_argument("--mean", type=float, default=0.5, help="特征归一化均值")
    ap.add_argument("--std", type=float, default=0.5, help="特征归一化标准差")
    ap.add_argument("--board", default=None, help="可选：ssh host，做版本体检")
    args = ap.parse_args()

    print(f"=== {TARGET_NAME} / ADLA3 转换（target_platform={TARGET_PLATFORM}）===\n")
    if args.board:
        print(json.dumps(check_board_runtime(args.board), ensure_ascii=False, indent=2))
        print()
    return convert(args.onnx, args.calib, args.out, args.dtype, args.mean, args.std)


if __name__ == "__main__":
    sys.exit(main())
