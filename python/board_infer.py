"""板端流式推理 + 分层延迟打点。

三层延迟必须分开记，否则"端到端 300 ms"没法优化：
    t_gate  门控（域 0）
    t_feat  Log-Mel（域 1）
    t_npu   推理（域 2）
    t_fsm   状态机
    t_e2e   窗末 → 最终决策（快路径不含 npu）

同时统计 npu_duty（NPU 实际调用窗数 / 总窗数），这是分级唤醒的核心证据。

用法（板端）::

    python board_infer.py --wav data/wav/glass.wav --model models/bcresnet_s_w8a8.adla
    python board_infer.py --wav data/wav/glass.wav --model ... --json bench/logs/e2e.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from audio_frontend import FrontendConfig, ReferenceFrontend

# 门控阈值：低于此能量（dBFS）直接判为静音，不进 NPU
GATE_DBFS = -55.0


def load_pcm(path: Path, sr: int) -> np.ndarray:
    import wave

    with wave.open(str(path), "rb") as w:
        if w.getframerate() != sr:
            raise ValueError(f"expect {sr} Hz, got {w.getframerate()}")
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        if w.getnchannels() > 1:
            data = data.reshape(-1, w.getnchannels()).mean(axis=1).astype(np.int16)
        return data


def dbfs(x: np.ndarray) -> float:
    if x.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean((x.astype(np.float64) / 32768.0) ** 2)))
    return 20.0 * np.log10(max(rms, 1e-12))


class DummyModel:
    """没有 .adla 时的占位：用线性投影模拟固定 shape 的前向开销。"""

    def __init__(self, n_mels: int, classes: int = 5):
        rng = np.random.default_rng(0)
        self.w = rng.standard_normal((n_mels, classes)).astype(np.float32) / n_mels
        self.classes = classes

    def __call__(self, feat: np.ndarray) -> np.ndarray:
        pooled = feat.mean(axis=1, keepdims=True)  # (n_mels, 1)
        logits = self.w.T @ pooled
        return (1.0 / (1.0 + np.exp(-logits))).ravel()


class AdlaModel:
    """真实 .adla 推理。加载失败时明确报错，不静默降级。"""

    def __init__(self, path: Path):
        try:
            import amlnn
        except ImportError as exc:
            raise RuntimeError(
                "板端需要 amlnn_edge_toolkit_lite。若只有 C++ 运行时，请用 cpp/ 下的实现。"
            ) from exc
        self.sess = amlnn.load(str(path))

    def __call__(self, feat: np.ndarray) -> np.ndarray:
        inp = feat[None, None, :, :].astype(np.float32)  # (1,1,n_mels,T)
        out = self.sess.run(inp)
        arr = np.asarray(out).ravel()
        if arr.min() < 0 or arr.max() > 1:
            arr = 1.0 / (1.0 + np.exp(-arr))
        return arr


def pct(v, q):
    return float(np.percentile(v, q)) if len(v) else float("nan")


def run(wav: Path, model_path: Path | None, cfg: FrontendConfig, gate: bool,
        realtime: bool = False) -> dict:
    pcm = load_pcm(wav, cfg.sr)
    frontend = ReferenceFrontend(cfg)
    model = AdlaModel(model_path) if model_path else DummyModel(cfg.n_mels)

    hop = cfg.hop_length
    win = cfg.win_length
    n_win = max(0, (pcm.size - win) // hop + 1)

    lat_gate, lat_feat, lat_npu, lat_fsm = [], [], [], []
    npu_calls = 0
    gated = 0
    scores = []

    t_stream0 = time.perf_counter()
    for i in range(n_win):
        seg = pcm[i * hop: i * hop + win]

        t0 = time.perf_counter()
        level = dbfs(seg)
        if gate and level < GATE_DBFS:
            gated += 1
            lat_gate.append((time.perf_counter() - t0) * 1000)
            scores.append(0.0)
            continue
        lat_gate.append((time.perf_counter() - t0) * 1000)

        t1 = time.perf_counter()
        feat = frontend.compute(seg)
        lat_feat.append((time.perf_counter() - t1) * 1000)

        t2 = time.perf_counter()
        prob = model(feat)
        lat_npu.append((time.perf_counter() - t2) * 1000)
        npu_calls += 1

        t3 = time.perf_counter()
        scores.append(float(prob.max()))
        lat_fsm.append((time.perf_counter() - t3) * 1000)

        if realtime:
            target = (i + 1) * hop / cfg.sr
            drift = target - (time.perf_counter() - t_stream0)
            if drift > 0:
                time.sleep(drift)

    wall_ms = (time.perf_counter() - t_stream0) * 1000
    audio_ms = n_win * hop / cfg.sr * 1000

    e2e_fast, e2e_confirm = [], []
    for i in range(n_win):
        gate_ms = lat_gate[i] if i < len(lat_gate) else 0.0
        if scores[i] > 0 and len(lat_feat) and len(lat_npu) and len(lat_fsm):
            # 走完整链路：门控 + 特征 + 推理 + 状态机
            idx = min(i, len(lat_feat) - 1)
            e2e_confirm.append(gate_ms + lat_feat[idx] + lat_npu[idx] + lat_fsm[idx])
        elif i < len(lat_gate):
            e2e_fast.append(gate_ms)

    return {
        "wav": wav.name,
        "windows": n_win,
        "gate_enabled": gate,
        "npu_calls": npu_calls,
        "gated_windows": gated,
        "npu_duty": round(npu_calls / n_win, 4) if n_win else 0.0,
        "latency_ms": {
            "gate": {"p50": round(pct(lat_gate, 50), 3), "p99": round(pct(lat_gate, 99), 3)},
            "feat": {"p50": round(pct(lat_feat, 50), 3), "p99": round(pct(lat_feat, 99), 3)},
            "npu": {"p50": round(pct(lat_npu, 50), 3), "p99": round(pct(lat_npu, 99), 3)},
            "fsm": {"p50": round(pct(lat_fsm, 50), 3), "p99": round(pct(lat_fsm, 99), 3)},
        },
        "e2e_ms": {
            "fastpath": {"n": len(e2e_fast), "p50": round(pct(e2e_fast, 50), 3),
                         "p99": round(pct(e2e_fast, 99), 3)},
            "confirm": {"n": len(e2e_confirm), "p50": round(pct(e2e_confirm, 50), 3),
                        "p99": round(pct(e2e_confirm, 99), 3)},
            "note": "两条曲线必须分开报：fastpath 是门控拒绝的多数窗，"
                    "confirm 才是有事件时用户感知的延迟；合成一个分位数会互相淹没。",
        },
        "rtf": round(wall_ms / audio_ms, 4) if audio_ms else None,
        "rtf_note": "realtime 模式下 rtf 应接近 1.0；离线模式 <1 表示快于实时",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=None, help=".adla；不传则用占位模型")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--n-mels", type=int, default=40)
    ap.add_argument("--no-gate", action="store_true", help="关掉门控做基线对照")
    ap.add_argument("--realtime", action="store_true", help="按 hop 限速模拟实时输入")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    cfg = FrontendConfig(sr=args.sr, n_mels=args.n_mels)
    res = run(args.wav, args.model, cfg, gate=not args.no_gate, realtime=args.realtime)

    print(json.dumps(res, ensure_ascii=False, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
