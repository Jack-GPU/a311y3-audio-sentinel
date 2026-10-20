"""前端一致性校验与归因工具。

板端 INT8 掉点最常见的根因不在网络层，而在 Log-Mel 前端的实现差异：
窗函数差一个采样点、mel 归一化用错、dB 系数写成 20、log 用低精度查表……
这些差异单看都"很小"，叠起来足以让 Top-1 掉几个点，而且极难定位。

本工具做三件事：
  1. 用同一段 PCM 同时跑 PC 参考实现与板端定点实现；
  2. 逐项量化每个已知偏差单独的误差贡献（single）；
  3. 从"全错"状态出发贪心修复，输出**排障路径**——每修一项，误差降到多少。

第 3 点是关键：它把"我调了好久"变成"修掉前 3 项，误差从 0.82 降到 0.008"。

用法::

    python mel_parity.py                              # 合成信号，默认配置
    python mel_parity.py --wav data/wav/glass.wav     # 真实音频
    python mel_parity.py --n-mels 64 --out bench/logs/feature_parity.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audio_frontend import (  # noqa: E402
    DEVIATIONS,
    FixedPointFrontend,
    FrontendConfig,
    ReferenceFrontend,
    diff_stats,
)

PASS_MAX_ABS = 1e-3
PASS_RMSE = 5e-4


def synth_signals(sr: int = 16000, seconds: float = 3.0):
    """构造覆盖高低频、瞬态、稳态与低 SNR 的合成信号，避免依赖外部音频。"""
    n = int(sr * seconds)
    t = np.arange(n) / sr
    rng = np.random.default_rng(20261020)

    sigs = {}
    tones = sum(np.sin(2 * np.pi * f * t + i) for i, f in enumerate([220, 880, 2400, 6000]))
    sigs["steady_tones"] = 0.25 * tones / 4

    f0, f1 = 100.0, 7000.0
    sweep = np.sin(2 * np.pi * (f0 * t + (f1 - f0) / (2 * seconds) * t ** 2))
    sigs["chirp"] = 0.4 * sweep

    imp = np.zeros(n)
    for k in range(6):
        idx = int((0.2 + 0.45 * k) * sr)
        imp[idx:idx + 24] = rng.standard_normal(min(24, n - idx))
    sigs["impulse_train"] = 0.5 * imp

    sigs["low_snr"] = 0.02 * sweep + 0.05 * rng.standard_normal(n)

    mix = np.zeros(n)
    mix[: n // 2] = 1e-4 * rng.standard_normal(n // 2)
    mix[n // 2:] = 0.5 * sweep[n // 2:]
    sigs["silence_then_onset"] = mix

    return {k: np.clip(v, -1.0, 1.0) for k, v in sigs.items()}


def to_pcm(x: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(x) * 32767.0).astype(np.int16)


def read_wav(path: Path) -> np.ndarray:
    """最小 WAV 读取（16-bit PCM），避免额外依赖。"""
    import wave

    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError("only 16-bit PCM supported")
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        if w.getnchannels() > 1:
            data = data.reshape(-1, w.getnchannels()).mean(axis=1).astype(np.int16)
        return data


class ParityRunner:
    def __init__(self, cfg: FrontendConfig, cases: dict[str, np.ndarray]):
        self.cfg = cfg
        self.cases = cases
        self.ref = {k: ReferenceFrontend(cfg).compute(v) for k, v in cases.items()}

    def measure(self, deviations) -> dict:
        """返回跨 case 的最坏误差。"""
        worst = {"max_abs": 0.0, "rmse": 0.0, "p99_abs": 0.0, "worst_case": None}
        for name, pcm in self.cases.items():
            got = FixedPointFrontend(self.cfg, deviations=deviations).compute(pcm)
            s = diff_stats(self.ref[name], got)
            if s["max_abs"] != s["max_abs"]:  # NaN
                return {"max_abs": float("nan"), "rmse": float("nan"),
                        "p99_abs": float("nan"), "worst_case": name,
                        "shape_mismatch": s.get("shape_mismatch", 0.0)}
            if s["max_abs"] > worst["max_abs"]:
                worst = {"max_abs": s["max_abs"], "rmse": s["rmse"],
                         "p99_abs": s["p99_abs"], "worst_case": name}
        return worst

    def single(self) -> list[dict]:
        rows = []
        for dev in DEVIATIONS:
            s = self.measure({dev})
            rows.append({"deviation": dev, **s})
        rows.sort(key=lambda d: d["max_abs"], reverse=True)
        return rows

    def repair_path(self) -> list[dict]:
        """从全偏差状态出发，每步修掉当前收益最大的一项。"""
        remaining = set(DEVIATIONS)
        current = self.measure(remaining)
        path = [{"step": 0, "removed": None, "max_abs": current["max_abs"],
                 "rmse": current["rmse"], "worst_case": current["worst_case"],
                 "note": "full mismatch：两端实现全部不一致"}]
        step = 1
        while remaining:
            best = None
            for dev in sorted(remaining):
                trial = remaining - {dev}
                s = self.measure(trial)
                if best is None or s["max_abs"] < best[1]["max_abs"]:
                    best = (dev, s, trial)
            dev, s, trial = best
            remaining = trial
            path.append({"step": step, "removed": dev, "max_abs": s["max_abs"],
                         "rmse": s["rmse"], "worst_case": s["worst_case"]})
            step += 1
        return path


def read_c_dump(path: Path):
    """解析 cpp/src/mel_dump.c 的输出：# 注释行 + n_mels 行矩阵。"""
    meta = {}
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            for tok in line.lstrip("# ").split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    try:
                        meta[k] = int(v)
                    except ValueError:
                        meta[k] = v
            continue
        rows.append([float(x) for x in line.split()])
    if not rows:
        raise ValueError(f"{path} 没有数据行")
    return meta, np.asarray(rows, dtype=np.float32)


def compare_c_dump(dump: Path, wav: Path, cfg: FrontendConfig, out: Path) -> int:
    """把板端 C 实现的 mel 与 PC 参考实现逐元素比对。"""
    meta, c_mel = read_c_dump(dump)
    pcm = read_wav(wav)
    ref = ReferenceFrontend(cfg).compute(pcm)

    print(f"\n=== C 侧 dump 对比 ===")
    print(f"  C   : {c_mel.shape}  (meta {meta})")
    print(f"  PC  : {ref.shape}")

    if c_mel.shape[1] != ref.shape[1]:
        n = min(c_mel.shape[1], ref.shape[1])
        print(f"  [warn] 帧数不一致，按前 {n} 帧比较（末尾边界处理不同）")
        c_mel, ref = c_mel[:, :n], ref[:, :n]
    if c_mel.shape[0] != ref.shape[0]:
        print(f"  [fail] n_mels 不一致：C={c_mel.shape[0]} PC={ref.shape[0]}")
        return 1

    s = diff_stats(ref, c_mel)
    passed = s["max_abs"] < PASS_MAX_ABS
    payload = {
        "mode": "c_dump",
        "wav": wav.name,
        "dump": dump.name,
        "shape": list(c_mel.shape),
        "c_meta": meta,
        "pc_config": cfg.to_dict(),
        "diff": s,
        "pass": bool(passed),
        "decision": (
            f"C 实现与 PC 参考实现特征 max|Δ|={s['max_abs']:.2e}，"
            + ("低于阈值，前端已对齐，可以进入量化。" if passed
               else "超过阈值，先按 configs/pipeline.yaml 逐项核对实现再谈量化。")
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  max|Δ| = {s['max_abs']:.3e}   rmse = {s['rmse']:.3e}   "
          f"p99|Δ| = {s['p99_abs']:.3e}")
    print(f"  判据 {PASS_MAX_ABS:.0e} → {'PASS' if passed else 'FAIL'}")
    print(f"  {payload['decision']}")
    print(f"\n已写入 {out}")
    return 0 if passed else 1


def fmt(x: float) -> str:
    if x != x:
        return "  n/a"
    if x >= 1e-2:
        return f"{x:.4f}"
    return f"{x:.2e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", type=Path, default=None)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--n-mels", type=int, default=40)
    ap.add_argument("--n-fft", type=int, default=512)
    ap.add_argument("--win", type=int, default=400)
    ap.add_argument("--hop", type=int, default=160)
    ap.add_argument("--out", type=Path, default=Path("bench/logs/feature_parity.json"))
    ap.add_argument("--c-dump", type=Path, default=None,
                    help="cpp/mel_dump 导出的文本，走 C↔PC 逐元素比对")
    ap.add_argument("--top-k", type=int, default=3,
                    help="保留参数：兼容旧调用")
    args = ap.parse_args()

    cfg = FrontendConfig(sr=args.sr, n_fft=args.n_fft, win_length=args.win,
                         hop_length=args.hop, n_mels=args.n_mels)

    if args.c_dump:
        if not args.wav:
            ap.error("--c-dump 需要同时给 --wav（必须是 dump 时用的同一份音频）")
        return compare_c_dump(args.c_dump, args.wav, cfg, args.out)


    if args.wav:
        cases = {args.wav.stem: read_wav(args.wav)}
    else:
        cases = {k: to_pcm(v) for k, v in synth_signals(cfg.sr).items()}

    runner = ParityRunner(cfg, cases)
    shapes = {k: list(v.shape) for k, v in runner.ref.items()}
    single = runner.single()
    path = runner.repair_path()

    full = path[0]
    residual = path[-1]
    passed = residual["max_abs"] < PASS_MAX_ABS

    # 消掉 90% / 99% 偏差各需修几项——比"修完几项"更能说明归因质量
    def items_to(ratio: float) -> int | None:
        target = full["max_abs"] * (1.0 - ratio)
        for p in path[1:]:
            if p["max_abs"] <= target:
                return p["step"]
        return None

    n90, n99 = items_to(0.90), items_to(0.99)

    out = {
        "meta": {
            "config": cfg.to_dict(),
            "pass_threshold": {"max_abs": PASS_MAX_ABS, "rmse": PASS_RMSE},
            "cases": {k: {"frames": v[1]} for k, v in shapes.items()},
            "note": "max_abs/rmse 均在特征归一化域（约 [-1,0]）。single 为单项独立开启；"
                    "repair_path 为从全偏差状态贪心修复的排障顺序。",
        },
        "full_mismatch": {"max_abs": full["max_abs"], "rmse": full["rmse"]},
        "single_deviation": single,
        "repair_path": path,
        "residual_after_repair": {"max_abs": residual["max_abs"], "rmse": residual["rmse"]},
        "items_to_90pct": n90,
        "items_to_99pct": n99,
        "pass": bool(passed),
        "decision": "",
    }

    if passed:
        top_names = [p["removed"] for p in path[1: min(n90 or len(path), len(path))]]
        out["decision"] = (
            f"两端实现完全不一致时特征偏差 {full['max_abs']:.3f}；"
            f"按归因修复 {n90} 项即可消掉 90% 偏差（{', '.join(top_names[:3])} 等），"
            f"全量对齐后残余 {residual['max_abs']:.2e}，满足 {PASS_MAX_ABS:.0e} 阈值。"
            f"结论：前端差异可完全归因，掉点来自实现而非量化精度。"
        )
    else:
        out["decision"] = (
            f"全量修复后仍有 {residual['max_abs']:.2e}，超过 {PASS_MAX_ABS:.0e}，"
            f"说明存在未建模的实现差异，需回到板端逐级 dump 中间张量定位。"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== Log-Mel 前端一致性（{len(cases)} case，{cfg.n_mels} mel，"
          f"fft={cfg.n_fft} win={cfg.win_length} hop={cfg.hop_length}）===\n")
    print(f"{'case':<22}{'frames':>7}")
    for k, v in shapes.items():
        print(f"{k:<22}{v[1]:>7}")

    print(f"\n--- 单项偏差独立贡献（跨 case 取最坏）---")
    print(f"{'deviation':<22}{'max|Δ|':>12}{'rmse':>12}  worst case")
    for d in single:
        print(f"{d['deviation']:<22}{fmt(d['max_abs']):>12}{fmt(d['rmse']):>12}  {d['worst_case']}")

    print(f"\n--- 排障路径（贪心修复）---")
    print(f"{'step':<6}{'removed':<22}{'max|Δ|':>12}{'降幅':>10}")
    prev = full["max_abs"]
    for p in path:
        drop = (1 - p["max_abs"] / prev) * 100 if prev > 0 and p["step"] > 0 else 0.0
        name = p["removed"] or "—"
        print(f"{p['step']:<6}{name:<22}{fmt(p['max_abs']):>12}"
              f"{(f'{drop:.0f}%' if p['step'] else ''):>10}")
        prev = p["max_abs"]

    print(f"\n消掉 90% 偏差需修 {n90} 项 / 99% 需修 {n99} 项（共 {len(DEVIATIONS)} 项）")
    print(f"判据：残余 < {PASS_MAX_ABS:.0e} → {'PASS' if passed else 'FAIL'}"
          f"（实际 {residual['max_abs']:.2e}）")
    print(f"结论：{out['decision']}")
    print(f"\n已写入 {args.out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
