"""ADLA3 算子可行性探测。

模型选型的顺序必须是「先探算子、再定结构」，反过来一定返工。
AMLNN 对 ONNX 的覆盖不是全集，音频模型常用的 GRU / LSTM / Conv1D
大概率落不到 NPU 上；一旦 fallback 到 CPU，「NPU 加速」就名存实亡。

本脚本做两件事：
  1. 输出 ADLA3 的算子风险清单（不依赖任何 SDK，随时可跑）；
  2. 装了 onnx / amlnn 时，为每个算子生成最小模型并尝试转换，实测支持情况。

用法::

    python probe_operators.py                    # 静态清单
    python probe_operators.py --emit-onnx        # 生成探测用最小 ONNX
    python probe_operators.py --try-convert      # 有 amlnn 时尝试真实转换
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

TARGET_PLATFORM = "007"  # AMLNN 平台表中 A311Y3 的 target id
TARGET_NAME = "A311Y3 / ADLA3"


@dataclass
class OpProbe:
    op: str
    usage: str
    risk: str          # low / medium / high
    fallback: str      # 不支持时怎么办
    in_audio_model: bool = True


# 依据 ADLA 系工具链对 CNN 友好、对 RNN 支持弱的普遍特性整理。
# risk 是"先验判断"，真值以 --try-convert 的实测为准。
OPS: List[OpProbe] = [
    OpProbe("Conv2D", "主干卷积", "low", "—"),
    OpProbe("DepthwiseConv2D", "BC-ResNet 的 separable conv", "low", "—"),
    OpProbe("BatchNormalization", "训练期算子", "low", "推理前折叠进 Conv"),
    OpProbe("Relu / Relu6", "激活", "low", "—"),
    OpProbe("Sigmoid", "注意力门控", "low", "—"),
    OpProbe("AvgPool2D", "时频下采样", "low", "—"),
    OpProbe("GlobalAveragePool", "分类头前池化", "low", "改 AvgPool + Reshape"),
    OpProbe("Gemm / MatMul", "全连接分类头", "medium", "改 Conv2D 1x1"),
    OpProbe("Softmax", "输出概率", "low", "移出模型，CPU 侧做"),
    OpProbe("Concat / Add / Mul", "残差与拼接", "low", "—"),
    OpProbe("Conv1D", "一维卷积音频模型常用", "medium", "改 Conv2D(kernel=(1,k))"),
    OpProbe("Transpose", "NCHW/NHWC 转换", "medium", "固化进前后处理"),
    OpProbe("Resize / Upsample", "UNet 类结构", "medium", "换 stride-transpose"),
    OpProbe("Slice / Split", "频谱裁剪", "medium", "改成常量切片"),
    OpProbe("GRU", "CRNN 时序建模", "high", "放弃，改纯 CNN（BC-ResNet）"),
    OpProbe("LSTM", "CRNN 时序建模", "high", "放弃，改纯 CNN"),
    OpProbe("LRN", "老式归一化", "high", "换 BatchNorm"),
    OpProbe("STFT / MelFilterBank", "把前端塞进模型", "high", "前端留在 CPU/DSP，别进图"),
]


def static_report() -> dict:
    counts = {"low": 0, "medium": 0, "high": 0}
    for o in OPS:
        counts[o.risk] += 1
    return {
        "target": {"platform_id": TARGET_PLATFORM, "name": TARGET_NAME},
        "summary": counts,
        "ops": [asdict(o) for o in OPS],
        "decision": (
            f"高风险算子 {counts['high']} 个，全部集中在时序建模（GRU/LSTM）与"
            f"图内前端（STFT/MelFilterBank）。因此模型选型锁定纯 CNN（BC-ResNet 系），"
            f"前端留在 CPU/DSP 侧——这是被部署约束倒推出来的结构决策，不是精度妥协。"
        ),
    }


def emit_onnx(out_dir: Path) -> int:
    """为每个算子生成最小 ONNX，供 --try-convert 逐个体检。"""
    try:
        import numpy as np
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError:
        print("[skip] 未安装 onnx，无法生成探测模型：pip install onnx")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    made = 0

    def save(name, nodes, inputs, outputs, inits=()):
        nonlocal made
        g = helper.make_graph(nodes, name, inputs, outputs, list(inits))
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
        m.ir_version = 8
        onnx.checker.check_model(m)
        onnx.save(m, str(out_dir / f"{name}.onnx"))
        made += 1

    def tensor(name, shape):
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)

    rng = np.random.default_rng(0)

    def w(name, shape):
        return numpy_helper.from_array(rng.standard_normal(shape).astype("float32") * 0.1, name)

    # Conv2D / DepthwiseConv2D
    save("conv2d",
         [helper.make_node("Conv", ["x", "w"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])],
         [tensor("x", [1, 8, 40, 32])], [tensor("y", [1, 16, 40, 32])],
         [w("w", [16, 8, 3, 3])])
    save("depthwise_conv2d",
         [helper.make_node("Conv", ["x", "w"], ["y"], kernel_shape=[3, 3],
                           pads=[1, 1, 1, 1], group=8)],
         [tensor("x", [1, 8, 40, 32])], [tensor("y", [1, 8, 40, 32])],
         [w("w", [8, 1, 3, 3])])
    # GlobalAveragePool
    save("global_avgpool",
         [helper.make_node("GlobalAveragePool", ["x"], ["y"])],
         [tensor("x", [1, 16, 8, 8])], [tensor("y", [1, 16, 1, 1])])
    # Gemm
    save("gemm",
         [helper.make_node("Gemm", ["x", "w", "b"], ["y"], transB=1)],
         [tensor("x", [1, 64])], [tensor("y", [1, 6])],
         [w("w", [6, 64]), numpy_helper.from_array(np.zeros(6, "float32"), "b")])
    # Conv1D（音频常用）
    save("conv1d",
         [helper.make_node("Conv", ["x", "w"], ["y"], kernel_shape=[3], pads=[1, 1])],
         [tensor("x", [1, 16, 100])], [tensor("y", [1, 16, 100])],
         [w("w", [16, 16, 3])])
    # GRU / LSTM（时序，高风险）
    save("gru",
         [helper.make_node("GRU", ["x", "w", "r", "b"], ["y"], hidden_size=16)],
         [tensor("x", [1, 20, 32])], [tensor("y", [1, 1, 16, 16])],
         [w("w", [1, 48, 16]), w("r", [1, 16, 16]),
          numpy_helper.from_array(np.zeros((1, 96), "float32"), "b")])
    save("lstm",
         [helper.make_node("LSTM", ["x", "w", "r", "b"], ["y"], hidden_size=16)],
         [tensor("x", [1, 20, 32])], [tensor("y", [1, 1, 16, 16])],
         [w("w", [1, 64, 16]), w("r", [1, 16, 16]),
          numpy_helper.from_array(np.zeros((1, 128), "float32"), "b")])

    print(f"[ok] 已生成 {made} 个探测模型 → {out_dir}")
    return 0


def try_convert(onnx_dir: Path, calib: Path | None, out_json: Path) -> int:
    try:
        import amlnn
    except ImportError:
        print("[skip] 未安装 amlnn-toolkit，无法实测转换。"
              "请在 PC 侧 toolkit 环境执行，或把本脚本拷到板端。")
        return 1

    if not onnx_dir.exists():
        print(f"[skip] {onnx_dir} 不存在，先跑 --emit-onnx")
        return 1

    results = []
    for f in sorted(onnx_dir.glob("*.onnx")):
        rec = {"op": f.stem, "converted": False, "error": None, "artifact": None}
        try:
            model = amlnn.load_onnx(str(f))
            kw = dict(normalization_mean=[[0.0]], normalization_std=[[1.0]],
                      quantized_dtype="w8a8", target_platform=f"PRODUCT_PID0XA{TARGET_PLATFORM}")
            if calib:
                kw["dataset"] = str(calib)
            amlnn.config(model, **kw)
            amlnn.compile(model)
            adla = f.with_suffix(".adla")
            amlnn.export_adla(model, str(adla))
            rec.update(converted=True, artifact=str(adla))
        except Exception as exc:  # noqa: BLE001 - 工具链异常类型不稳定，统一记录
            rec["error"] = f"{type(exc).__name__}: {exc}"
        results.append(rec)
        print(f"  {rec['op']:<20}{'OK' if rec['converted'] else 'FAIL'}  {rec['error'] or ''}")

    payload = {"target": TARGET_NAME, "platform_id": TARGET_PLATFORM,
               "results": results,
               "decision": "以本表为准更新模型结构；未通过项不得出现在主干中。"}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {out_json}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-onnx", action="store_true")
    ap.add_argument("--try-convert", action="store_true")
    ap.add_argument("--onnx-dir", type=Path, default=Path("models/probe_onnx"))
    ap.add_argument("--calib", type=Path, default=None,
                    help="校准数据列表（每行一个 wav 路径）")
    ap.add_argument("--out", type=Path, default=Path("bench/logs/operator_probe.json"))
    args = ap.parse_args()

    if args.try_convert:
        return try_convert(args.onnx_dir, args.calib, args.out)

    if args.emit_onnx:
        emit_onnx(args.onnx_dir)

    rep = static_report()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== ADLA3 算子风险清单（target_platform={TARGET_PLATFORM}，{TARGET_NAME}）===\n")
    print(f"{'op':<24}{'risk':<10}{'audio':<7}fallback")
    for o in OPS:
        print(f"{o.op:<24}{o.risk:<10}{'yes' if o.in_audio_model else 'no':<7}{o.fallback}")

    s = rep["summary"]
    print(f"\n高风险 {s['high']} / 中 {s['medium']} / 低 {s['low']}")
    print(f"结论：{rep['decision']}")
    print(f"\n已写入 {args.out}")
    print("下一步：--emit-onnx 生成探测模型，在 toolkit 环境跑 --try-convert 拿实测真值。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
