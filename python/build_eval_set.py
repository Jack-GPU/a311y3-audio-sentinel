"""生成合成评测集（无版权风险，可入库复现）。

不做"网上下载数据集"的路线：版权不清、口径不可控、复现时别人拿不到。
这里用物理上可解释的合成方式造出正类与硬负类，重点是**负类**——
人声、电视、音乐才是误报主战场，只测正类的项目说明没做过产品。

用法::

    python build_eval_set.py --out data/wav --cases data/eval/cases.jsonl
"""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np

SR = 16000


def write_wav(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(x, -1.0, 1.0)
    pcm = np.round(pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def env_ad(n: int, attack: int, decay: int) -> np.ndarray:
    e = np.ones(n)
    a = min(attack, n)
    e[:a] = np.linspace(0, 1, a)
    d = min(decay, n)
    e[-d:] = np.linspace(1, 0, d) ** 2
    return e


def glass_break(dur=0.8, seed=1):
    """玻璃破碎：极短上升 + 高频密集衰减 + 少量低频撞击。"""
    rng = np.random.default_rng(seed)
    n = int(SR * dur)
    t = np.arange(n) / SR
    noise = rng.standard_normal(n)
    # 高频湍流
    hp = np.diff(np.concatenate([[0.0], noise])) * 0.6
    burst = hp * np.exp(-t * 18) * 0.9
    # 若干谐振分量
    res = sum(a * np.sin(2 * np.pi * f * t) * np.exp(-t * 22)
              for f, a in [(3200, .18), (4700, .12), (6100, .08)])
    thump = 0.35 * np.sin(2 * np.pi * 120 * t) * np.exp(-t * 30)
    return 0.9 * (burst + res + thump) * env_ad(n, 8, n // 3)


def dog_bark(dur=2.0, seed=2):
    """犬吠：200-700 Hz 基频 + 谐波 + 快速包络起伏。"""
    rng = np.random.default_rng(seed)
    n = int(SR * dur)
    t = np.arange(n) / SR
    f0 = 380 + 120 * np.sin(2 * np.pi * 3.2 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    harm = sum((0.6 ** k) * np.sin((k + 1) * phase) for k in range(5))
    rough = 1 + 0.25 * rng.standard_normal(n)
    bark_env = np.zeros(n)
    for start in (0.1, 0.55, 1.05, 1.55):
        i = int(start * SR)
        seg = min(int(0.22 * SR), n - i)
        if seg > 0:
            bark_env[i:i + seg] = env_ad(seg, int(0.01 * SR), int(0.2 * SR))
    return 0.45 * harm * rough * bark_env


def baby_cry(dur=2.0, seed=3):
    """婴儿哭：300-500 Hz 基频滑动 + 强颤音 + 共振峰。"""
    n = int(SR * dur)
    t = np.arange(n) / SR
    f0 = 420 + 90 * np.sin(2 * np.pi * 1.1 * t) + 40 * np.sin(2 * np.pi * 5.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    sig = np.sin(phase) + 0.45 * np.sin(2 * phase) + 0.2 * np.sin(3 * phase)
    formant = 1 + 0.4 * np.sin(2 * np.pi * 900 * t)
    return 0.4 * sig * formant * env_ad(n, int(0.15 * SR), int(0.3 * SR))


def speech_like(dur=3.0, seed=4):
    """人声硬负类：共振峰 + 音节包络 + 噪声成分（最容易误报的一类）。"""
    rng = np.random.default_rng(seed)
    n = int(SR * dur)
    t = np.arange(n) / SR
    f0 = 130 + 25 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    src = sum((0.5 ** k) * np.sin((k + 1) * phase) for k in range(12))
    src += 0.15 * rng.standard_normal(n)
    syll = np.zeros(n)
    pos = 0
    while pos < n:
        L = int(rng.uniform(0.12, 0.3) * SR)
        seg = min(L, n - pos)
        syll[pos:pos + seg] = env_ad(seg, int(0.03 * SR), int(0.06 * SR)) * rng.uniform(0.5, 1.0)
        pos += seg + int(rng.uniform(0.03, 0.12) * SR)
    formants = 1 + 0.6 * np.sin(2 * np.pi * 700 * t) + 0.3 * np.sin(2 * np.pi * 1200 * t)
    return 0.3 * src * syll * formants


def music_like(dur=3.0, seed=5):
    """音乐硬负类：稳定和弦 + 节拍。"""
    rng = np.random.default_rng(seed)
    n = int(SR * dur)
    t = np.arange(n) / SR
    chord = sum(np.sin(2 * np.pi * f * t) for f in (261.6, 329.6, 392.0, 523.3)) / 4
    beat = np.zeros(n)
    step = int(0.5 * SR)
    for i in range(0, n, step):
        seg = min(int(0.08 * SR), n - i)
        beat[i:i + seg] = np.exp(-np.linspace(0, 6, seg))
    return 0.28 * chord * (0.7 + 0.5 * beat) + 0.01 * rng.standard_normal(n)


def tv_noise(dur=3.0, seed=6):
    rng = np.random.default_rng(seed)
    n = int(SR * dur)
    t = np.arange(n) / SR
    base = 0.06 * rng.standard_normal(n)
    # 电视机常见的行频啸叫与低频轰鸣
    whine = 0.02 * np.sin(2 * np.pi * 7800 * t)
    hum = 0.03 * np.sin(2 * np.pi * 100 * t)
    return base + whine + hum


def silence(dur=3.0, seed=7):
    rng = np.random.default_rng(seed)
    return 0.0005 * rng.standard_normal(int(SR * dur))


def low_snr_glass(dur=1.0, seed=8, snr_db=5.0):
    """低 SNR 玻璃破碎：检验工作点是否脆。"""
    clean = glass_break(dur, seed)
    rng = np.random.default_rng(seed + 100)
    noise = rng.standard_normal(clean.size)
    sp = np.mean(clean ** 2)
    npow = sp / (10 ** (snr_db / 10))
    noise *= np.sqrt(npow / max(np.mean(noise ** 2), 1e-12))
    return clean + noise


def repeat_bark(dur=30.0, seed=9):
    """连续犬吠 30 s：检验冷却是否抑刷屏。"""
    out = np.zeros(int(SR * dur))
    unit = dog_bark(0.9, seed)
    pos = 0
    while pos + unit.size < out.size:
        out[pos:pos + unit.size] += unit * 0.8
        pos += int(SR * 1.2)
    return out


EVENTS = [
    ("glass_break_clean", "glass_break", True, "glass_break", "miss_sensitive", "positive_clean", glass_break),
    ("glass_break_lowsnr", "glass_break", True, "glass_break", "miss_sensitive", "positive_noisy",
     lambda: low_snr_glass(1.0, 81, 5.0)),
    ("glass_break_short", "glass_break", True, "glass_break", "miss_sensitive", "positive_short",
     lambda: glass_break(0.35, 82)),
    ("dog_bark_clean", "dog_bark", True, "dog_bark", "fa_sensitive", "positive_clean", dog_bark),
    ("dog_bark_repeat30s", "dog_bark", True, "dog_bark", "fa_sensitive", "burst_repeat",
     lambda: repeat_bark(30.0, 91)),
    ("baby_cry_clean", "baby_cry", True, "baby_cry", "neutral", "positive_clean", baby_cry),
    ("neg_speech", None, False, None, "neutral", "negative_speech", speech_like),
    ("neg_music", None, False, None, "neutral", "negative_music", music_like),
    ("neg_tv", None, False, None, "neutral", "negative_tv", tv_noise),
    ("neg_silence", None, False, None, "neutral", "long_silence", silence),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/wav"))
    ap.add_argument("--cases", type=Path, default=Path("data/eval/cases.jsonl"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    args.cases.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, expect_event, expect_alarm, event_id, cost_class, tag, fn in EVENTS:
        sig = fn()
        path = args.out / f"{name}.wav"
        write_wav(path, sig)
        rows.append({
            "case_id": name,
            "wav": str(path).replace("\\", "/"),
            "dur_s": round(sig.size / SR, 2),
            "expect_event": expect_event,
            "expect_alarm": expect_alarm,
            "event_id": event_id,
            "cost_class": cost_class,
            "tag": tag,
        })
        print(f"  {name:<24}{sig.size / SR:6.2f}s  {tag}")

    with args.cases.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_pos = sum(1 for r in rows if r["expect_alarm"])
    print(f"\n生成 {len(rows)} 条（正类 {n_pos} / 负类 {len(rows) - n_pos}）")
    print(f"音频 → {args.out}")
    print(f"清单 → {args.cases}")
    print("\n注意：合成集只用于打通链路与回归，正式结论需补真实录音（现场噪声分布不同）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
