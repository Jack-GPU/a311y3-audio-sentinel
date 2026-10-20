"""板端功耗账本：把"常开"折算成可比较的数字。

口径纪律（本项目最容易说错的地方）：

* 开发板是 12V/3A 整板（DDR + PMIC + HDMI + PHY 全在），整板功耗 2-4 W 量级，
  **不存在 mW 级整机常开**。宣称 mW 级常开的是成品板，不是开发板。
* 所以只报两种口径，且必须说明是哪种：
    A. NPU/CPU 活跃占空比（cpuidle residency + 调用计数推导）→ 折算平均功耗
    B. 整板平均功耗（INA226 串 12V 输入实测）
* 两种口径都要能和"全程 NPU 常开"的基线对比，否则数字没有意义。

用法::

    python board_power_budget.py --mode residency --secs 60
    python board_power_budget.py --mode ina226 --device /dev/i2c-1 --addr 0x40 --secs 60
    python board_power_budget.py --mode residency --out bench/logs/power_budget.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

CPUIDLE_ROOT = Path("/sys/devices/system/cpu")


def read_cpuidle() -> dict:
    """读各核各 C-state 的 usage/time。"""
    out = {}
    if not CPUIDLE_ROOT.exists():
        return out
    for cpu in sorted(CPUIDLE_ROOT.glob("cpu[0-9]*")):
        states = {}
        for st in sorted((cpu / "cpuidle").glob("state*")):
            try:
                name = (st / "name").read_text().strip()
                usage = int((st / "usage").read_text().strip())
                us = int((st / "time").read_text().strip())
                states[name] = {"usage": usage, "time_us": us}
            except (OSError, ValueError):
                continue
        if states:
            out[cpu.name] = states
    return out


def read_cpufreq() -> dict:
    out = {}
    for cpu in sorted(CPUIDLE_ROOT.glob("cpu[0-9]*")):
        base = cpu / "cpufreq"
        if not base.exists():
            continue
        try:
            out[cpu.name] = {
                "cur_khz": int((base / "scaling_cur_freq").read_text()),
                "min_khz": int((base / "scaling_min_freq").read_text()),
                "max_khz": int((base / "scaling_max_freq").read_text()),
                "governor": (base / "scaling_governor").read_text().strip(),
            }
        except (OSError, ValueError):
            continue
    return out


def residency_window(secs: float, interval: float = 1.0) -> dict:
    """采两个时间点，算窗口内各核深睡占比。"""
    before = read_cpuidle()
    if not before:
        return {"available": False,
                "reason": "no cpuidle nodes (kernel 未开 CONFIG_CPU_IDLE 或容器内不可见)"}

    t0 = time.time()
    time.sleep(secs)
    after = read_cpuidle()
    wall_us = (time.time() - t0) * 1e6

    per_cpu = {}
    for cpu, states in after.items():
        if cpu not in before:
            continue
        total_busy_us = 0.0
        detail = {}
        for name, s in states.items():
            if name in ("poll", "POLL"):
                continue
            prev = before[cpu].get(name, {"usage": 0, "time_us": 0})
            d_usage = s["usage"] - prev["usage"]
            d_us = s["time_us"] - prev["time_us"]
            if d_usage > 0:
                detail[name] = {"entries": d_usage, "ms": round(d_us / 1000.0, 2)}
                total_busy_us += d_us
        per_cpu[cpu] = {
            "deep_sleep_ms": round(total_busy_us / 1000.0, 2),
            "wall_ms": round(wall_us / 1000.0, 2),
            "deep_sleep_ratio": round(min(1.0, total_busy_us / wall_us), 4),
            "states": detail,
        }

    ratios = [v["deep_sleep_ratio"] for v in per_cpu.values() if v["wall_ms"] > 0]
    return {
        "available": True,
        "window_s": secs,
        "per_cpu": per_cpu,
        "mean_deep_sleep_ratio": round(sum(ratios) / len(ratios), 4) if ratios else 0.0,
        "mean_active_ratio": round(1 - (sum(ratios) / len(ratios)), 4) if ratios else 1.0,
        "note": "deep_sleep_ratio 只统计非 poll 状态；A55 深睡才是常开功耗的主要杠杆。",
    }


def ina226_window(device: str, addr: str, secs: float, shunt_ohm: float,
                  bus_v: float = 12.0) -> dict:
    """INA226 采样：读 shunt 电压与总线电压，按 1 kHz 估平均功率。"""
    try:
        from smbus2 import SMBus  # type: ignore
    except ImportError:
        return {"available": False, "reason": "pip install smbus2"}

    REG_SHUNT_V = 0x01   # LSB = 2.5 uV
    REG_BUS_V = 0x02     # LSB = 1.25 mV

    samples = []
    t_end = time.time() + secs
    with SMBus(int(device.split("-")[-1])) as bus:
        a = int(addr, 0)
        while time.time() < t_end:
            try:
                raw_s = bus.read_word_data(a, REG_SHUNT_V)
                raw_b = bus.read_word_data(a, REG_BUS_V)
                raw_s = ((raw_s & 0xFF) << 8) | (raw_s >> 8)
                raw_b = ((raw_b & 0xFF) << 8) | (raw_b >> 8)
                if raw_s > 32767:
                    raw_s -= 65536
                shunt_v = raw_s * 2.5e-6
                bus_v = (raw_b >> 3) * 1.25e-3
                i = shunt_v / shunt_ohm
                samples.append(bus_v * i)
            except OSError:
                pass
            time.sleep(0.001)

    if not samples:
        return {"available": False, "reason": "no samples"}
    samples.sort()
    n = len(samples)
    return {
        "available": True,
        "samples": n,
        "mean_w": round(sum(samples) / n, 3),
        "p50_w": round(samples[n // 2], 3),
        "p99_w": round(samples[min(n - 1, int(n * 0.99))], 3),
        "min_w": round(samples[0], 3),
        "max_w": round(samples[-1], 3),
        "bus_v_assumed": bus_v,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["residency", "ina226"], default="residency")
    ap.add_argument("--secs", type=float, default=60.0)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--device", default="/dev/i2c-1")
    ap.add_argument("--addr", default="0x40")
    ap.add_argument("--shunt-ohm", type=float, default=0.002)
    ap.add_argument("--out", type=Path, default=Path("bench/logs/power_residency.json"))
    ap.add_argument("--label", default="", help="标记：baseline / gated / cascade")
    args = ap.parse_args()

    if args.mode == "residency":
        res = residency_window(args.secs, args.interval)
        res["cpufreq"] = read_cpufreq()
    else:
        res = ina226_window(args.device, args.addr, args.secs, args.shunt_ohm)

    payload = {"label": args.label, "mode": args.mode, "result": res,
               "caveat": "开发板整板含 DDR/PMIC/HDMI/PHY，不可用本数据宣称 mW 级常开。"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if res.get("available"):
        if args.mode == "residency":
            print(f"\n平均深睡占比 {res['mean_deep_sleep_ratio']:.2%} → "
                  f"活跃占空比 {res['mean_active_ratio']:.2%}")
        else:
            print(f"\n整板平均 {res['mean_w']} W（P99 {res['p99_w']} W）")
    print(f"已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
