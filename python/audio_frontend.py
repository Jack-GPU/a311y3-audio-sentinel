"""Log-Mel 前端：PC 参考实现 + 板端定点模拟实现。

两份实现必须输出一致的特征。实践中"不一致"是 INT8 掉点的头号来源，
所以这里把每一处可能不同的实现选择都做成独立开关（``DEVIATIONS``），
供 ``mel_parity.py`` 逐项打开、量化每一项对特征的误差贡献。

板端真机实现见 ``cpp/src/mel_frontend.c``（Q15 定点）。本文件中的
``FixedPointFrontend`` 是它的数值等价模拟，用于在没有板子时先做归因。

约定
----
* 输入 PCM 一律 int16，单声道，16 kHz。
* 参考实现内部用 float32，最后按 ``FEATURE_NORM`` 归一化到约 [-1, 0]。
* 幅值→dB 用 ``10*log10(power)``；特征归一化 ``clip(db, -top_db, 0) / top_db``。
"""

from __future__ import annotations

from dataclasses import dataclass, replace, asdict
from typing import Dict, List

import numpy as np

# 逐项"板端实现可能长这样"的偏差开关。mel_parity.py 按此顺序逐项叠加。
DEVIATIONS: List[str] = [
    "window_symmetric",   # 对称窗而非周期窗（差一个采样点）
    "pad_zero",           # 零填充而非 reflect
    "preemph_skip",       # 漏掉预加重
    "mel_htk_norm",       # HTK 面积归一而非 Slaney
    "log_20",             # 20*log10 而非 10*log10
    "top_db_missing",     # 不做 top_db 截断（动态范围失控）
    "log_lut",            # log 用定点查表（步进近似）
    "filterbank_q15",     # 滤波器系数量化到 Q15
    "feature_int8",       # 特征量化到 int8
]


@dataclass(frozen=True)
class FrontendConfig:
    sr: int = 16000
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 40
    fmin: float = 20.0
    fmax: float | None = 7600.0
    preemph: float = 0.97
    window: str = "hann_periodic"
    pad_mode: str = "reflect"
    center: bool = True
    mel_norm: str = "slaney"
    log_mult: float = 10.0
    top_db: float = 80.0
    eps: float = 1e-10
    feature_norm: str = "db_over_top_db"

    def to_dict(self) -> Dict:
        return asdict(self)


# --------------------------------------------------------------------------
# mel scale
# --------------------------------------------------------------------------

def hz_to_mel(freq, norm: str):
    freq = np.asarray(freq, dtype=np.float64)
    if norm == "htk":
        return 2595.0 * np.log10(1.0 + freq / 700.0)
    # Slaney (librosa 默认): 1 kHz 以下线性，以上对数
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    mel = freq / f_sp
    log_part = min_log_mel + np.log(np.maximum(freq, 1e-10) / min_log_hz) / logstep
    return np.where(freq >= min_log_hz, log_part, mel)


def mel_to_hz(mel, norm: str):
    mel = np.asarray(mel, dtype=np.float64)
    if norm == "htk":
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    freq = f_sp * mel
    log_part = min_log_hz * np.exp(logstep * (mel - min_log_mel))
    return np.where(mel >= min_log_mel, log_part, freq)


def mel_filterbank(cfg: FrontendConfig, norm: str) -> np.ndarray:
    """返回 (n_mels, 1 + n_fft//2) 的三角滤波器组。"""
    fmax = cfg.fmax if cfg.fmax is not None else cfg.sr / 2.0
    fft_freqs = np.linspace(0.0, cfg.sr / 2.0, 1 + cfg.n_fft // 2)

    mel_pts = np.linspace(
        hz_to_mel(cfg.fmin, norm), hz_to_mel(fmax, norm), cfg.n_mels + 2
    )
    hz_pts = mel_to_hz(mel_pts, norm)

    fb = np.zeros((cfg.n_mels, fft_freqs.size), dtype=np.float64)
    for m in range(cfg.n_mels):
        lo, ce, hi = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
        rising = (fft_freqs - lo) / max(ce - lo, 1e-12)
        falling = (hi - fft_freqs) / max(hi - ce, 1e-12)
        fb[m] = np.maximum(0.0, np.minimum(rising, falling))

    if norm == "slaney":
        # 每个三角形按面积（Hz 宽度）归一，使低频带不被放大
        widths = hz_pts[2:] - hz_pts[:-2]
        fb *= (2.0 / np.maximum(widths, 1e-12))[:, None]
    return fb


def make_window(cfg: FrontendConfig) -> np.ndarray:
    if cfg.window == "hann_periodic":
        return np.hanning(cfg.win_length + 1)[:-1]
    if cfg.window == "hann_symmetric":
        return np.hanning(cfg.win_length)
    if cfg.window == "hamming_periodic":
        return np.hamming(cfg.win_length + 1)[:-1]
    raise ValueError(f"unknown window: {cfg.window}")


# --------------------------------------------------------------------------
# 参考实现（float32）
# --------------------------------------------------------------------------

def _frame(pcm: np.ndarray, cfg: FrontendConfig, pad_mode: str, center: bool):
    x = np.asarray(pcm, dtype=np.float64) / 32768.0
    if center:
        pad = cfg.n_fft // 2
        if pad_mode == "reflect":
            x = np.pad(x, pad, mode="reflect")
        else:
            x = np.pad(x, pad, mode="constant")
    n_frames = 1 + max(0, (x.size - cfg.win_length) // cfg.hop_length)
    if n_frames <= 0:
        return np.zeros((0, cfg.win_length)), 0
    idx = np.arange(cfg.win_length)[None, :] + cfg.hop_length * np.arange(n_frames)[:, None]
    frames = x[idx]
    return frames, n_frames


def _pad_to_fft(frames: np.ndarray, cfg: FrontendConfig) -> np.ndarray:
    if cfg.n_fft > cfg.win_length:
        return np.pad(frames, ((0, 0), (0, cfg.n_fft - cfg.win_length)))
    return frames


def _preemph(x: np.ndarray, coef: float) -> np.ndarray:
    if coef <= 0.0:
        return x
    y = np.empty_like(x)
    y[..., 0] = x[..., 0]
    y[..., 1:] = x[..., 1:] - coef * x[..., :-1]
    return y


def _to_db(power: np.ndarray, mult: float, eps: float, top_db: float) -> np.ndarray:
    db = mult * np.log10(np.maximum(power, eps))
    peak = db.max() if db.size else 0.0
    return np.maximum(db, peak - top_db)


def _normalize(db: np.ndarray, cfg: FrontendConfig) -> np.ndarray:
    if cfg.feature_norm == "db_over_top_db":
        return np.clip(db, -cfg.top_db, 0.0) / cfg.top_db
    if cfg.feature_norm == "minmax":
        lo, hi = float(db.min()), float(db.max())
        return (db - lo) / max(hi - lo, 1e-12)
    return db


class ReferenceFrontend:
    """PC 侧参考实现。板端必须对齐到这一份。"""

    def __init__(self, cfg: FrontendConfig):
        self.cfg = cfg
        self.fb = mel_filterbank(cfg, cfg.mel_norm)
        self.window = make_window(cfg)

    def compute(self, pcm: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        frames, n = _frame(pcm, cfg, cfg.pad_mode, cfg.center)
        if n == 0:
            return np.zeros((cfg.n_mels, 0), dtype=np.float32)
        frames = _preemph(frames, cfg.preemph)
        frames = frames * self.window[None, :]
        frames = _pad_to_fft(frames, cfg)
        spec = np.fft.rfft(frames, n=cfg.n_fft, axis=1)
        power = (spec.real ** 2 + spec.imag ** 2).T  # (bins, frames)
        mel = self.fb @ power
        db = _to_db(mel, cfg.log_mult, cfg.eps, cfg.top_db)
        return _normalize(db, cfg).astype(np.float32)


# --------------------------------------------------------------------------
# 板端定点模拟
# --------------------------------------------------------------------------

_Q15 = 32768.0


def _q15(x: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(x) * _Q15).clip(-_Q15, _Q15 - 1) / _Q15


class FixedPointFrontend:
    """板端 Q15 定点实现的数值模拟。

    ``deviations`` 给出这道"实现"相对参考实现犯了哪些错——真实项目里
    这些错通常不是故意的，而是 coder 各自理解不同导致的。
    """

    def __init__(self, cfg: FrontendConfig, deviations=()):
        self.cfg = replace(cfg)
        self.dev = set(deviations)
        if "window_symmetric" in self.dev:
            self.cfg = replace(self.cfg, window="hann_symmetric")
        if "mel_htk_norm" in self.dev:
            self.cfg = replace(self.cfg, mel_norm="htk")
        if "log_20" in self.dev:
            self.cfg = replace(self.cfg, log_mult=20.0)
        if "top_db_missing" in self.dev:
            self.cfg = replace(self.cfg, top_db=1e9)
        if "pad_zero" in self.dev:
            self.cfg = replace(self.cfg, pad_mode="constant")
        if "preemph_skip" in self.dev:
            self.cfg = replace(self.cfg, preemph=0.0)

        self.fb = mel_filterbank(self.cfg, self.cfg.mel_norm)
        if "filterbank_q15" in self.dev:
            self.fb = _q15(self.fb)
        self.window = make_window(self.cfg)

    def compute(self, pcm: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        frames, n = _frame(pcm, cfg, cfg.pad_mode, cfg.center)
        if n == 0:
            return np.zeros((cfg.n_mels, 0), dtype=np.float32)
        frames = _preemph(frames, cfg.preemph)
        frames = frames * self.window[None, :]
        frames = _pad_to_fft(frames, cfg)
        spec = np.fft.rfft(frames, n=cfg.n_fft, axis=1)
        power = (spec.real ** 2 + spec.imag ** 2).T
        mel = self.fb @ power

        db = cfg.log_mult * np.log10(np.maximum(mel, cfg.eps))
        peak = db.max() if db.size else 0.0
        db = np.maximum(db, peak - cfg.top_db)

        if "log_lut" in self.dev:
            # 定点查表：log 结果按 0.5 dB 步进取整，省掉实时 log10
            step = 0.5
            db = np.round(db / step) * step

        if "top_db_missing" in self.dev:
            feat = db / cfg.top_db  # 不裁剪，动态范围泄漏到特征
        else:
            feat = np.clip(db, -cfg.top_db, 0.0) / cfg.top_db

        if "feature_int8" in self.dev:
            scale = 1.0 / 127.0
            feat = np.round(feat / scale).clip(-127, 127) * scale
        return feat.astype(np.float32)


# --------------------------------------------------------------------------
# 误差度量
# --------------------------------------------------------------------------

def diff_stats(ref: np.ndarray, got: np.ndarray) -> Dict[str, float]:
    if ref.shape != got.shape:
        return {"shape_mismatch": 1.0, "max_abs": float("nan"), "rmse": float("nan")}
    d = np.abs(ref.astype(np.float64) - got.astype(np.float64))
    return {
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "rmse": float(np.sqrt((d ** 2).mean())),
        "p99_abs": float(np.percentile(d, 99)),
    }
