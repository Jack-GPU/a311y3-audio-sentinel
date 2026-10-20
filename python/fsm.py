"""按事件代价配置的非对称事件状态机。

同一个阈值不可能同时服务"怕漏检"和"怕误报"两类事件：
玻璃破碎漏一次是事故，犬吠误报一次是投诉。所以阈值必须按事件配置，
并且要有连续命中与冷却，否则单个误判就会刷屏。

状态流转::

    idle ──hit──> candidate ──连续 k 帧──> active ──告警──> cooldown
      ↑                │                                        │
      └─────miss───────┘                                        │
      └──────────────────── 冷却到期 ───────────────────────────┘

每个 (event, frame_score) 走一次 step()，emit 只返回"应当上报"的事件。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class EventPolicy:
    """单个事件的判决策略。"""

    event_id: str
    threshold: float          # 单帧触发阈值的下限
    min_consecutive: int      # 连续命中帧数
    cooldown_frames: int      # 告警后静默帧数
    cost_class: str           # miss_sensitive / fa_sensitive / neutral

    def describe(self) -> str:
        return (f"{self.event_id}: thr>={self.threshold} k={self.min_consecutive} "
                f"cd={self.cooldown_frames} ({self.cost_class})")


DEFAULT_POLICIES: Dict[str, EventPolicy] = {
    # 怕漏检：低阈值、短连续帧
    "glass_break": EventPolicy("glass_break", 0.35, 2, 15, "miss_sensitive"),
    "smoke_alarm": EventPolicy("smoke_alarm", 0.40, 2, 15, "miss_sensitive"),
    # 中性
    "baby_cry": EventPolicy("baby_cry", 0.55, 3, 30, "neutral"),
    # 怕误报：高阈值、长冷却
    "dog_bark": EventPolicy("dog_bark", 0.75, 4, 90, "fa_sensitive"),
}


@dataclass
class EventState:
    state: str = "idle"
    streak: int = 0
    cooldown_left: int = 0
    alarms: int = 0
    suppressed_by_cooldown: int = 0


@dataclass
class Decision:
    event_id: str
    fired: bool
    reason: str
    score: float
    state_after: str


class AsymmetricFSM:
    """多事件共享一个状态机，但每个事件独立状态与策略。"""

    def __init__(self, policies: Optional[Dict[str, EventPolicy]] = None):
        self.policies = policies or DEFAULT_POLICIES
        self.states: Dict[str, EventState] = {k: EventState() for k in self.policies}
        self.history: List[Decision] = []

    # -- 单事件推进 ------------------------------------------------------
    def step(self, event_id: str, score: float) -> Decision:
        pol = self.policies[event_id]
        st = self.states[event_id]

        if st.cooldown_left > 0:
            st.cooldown_left -= 1
            st.state = "cooldown"
            if score >= pol.threshold:
                st.suppressed_by_cooldown += 1
                return self._record(Decision(event_id, False, "REJECT_COOLDOWN", score, st.state))
            if st.cooldown_left == 0:
                st.state, st.streak = "idle", 0
            return self._record(Decision(event_id, False, "REJECT_COOLDOWN", score, st.state))

        if score < pol.threshold:
            st.state, st.streak = "idle", 0
            return self._record(Decision(event_id, False, "REJECT_LOW_SCORE", score, st.state))

        st.streak += 1
        st.state = "candidate"
        if st.streak >= pol.min_consecutive:
            st.alarms += 1
            st.state = "cooldown"
            st.cooldown_left = pol.cooldown_frames
            st.streak = 0
            return self._record(Decision(event_id, True, "ALARM", score, st.state))
        return self._record(Decision(event_id, False, "CANDIDATE", score, st.state))

    def step_frame(self, scores: Dict[str, float]) -> List[Decision]:
        """一帧内评估所有事件。"""
        return [self.step(k, v) for k, v in scores.items() if k in self.policies]

    def _record(self, d: Decision) -> Decision:
        self.history.append(d)
        return d

    # -- 观测 ------------------------------------------------------------
    def summary(self) -> Dict[str, dict]:
        return {
            k: {"state": v.state, "alarms": v.alarms,
                "suppressed_by_cooldown": v.suppressed_by_cooldown}
            for k, v in self.states.items()
        }


def demo() -> None:
    """跑一段人为构造的分数序列，验证非对称策略是否按预期生效。"""
    fsm = AsymmetricFSM()

    print("\n=== 非对称状态机自检 ===\n")
    for p in fsm.policies.values():
        print("  " + p.describe())

    print("\n[场景 1] 犬吠：连续 20 帧 0.80（超过阈值但需 4 连击）")
    fired = 0
    for _ in range(20):
        fired += int(fsm.step("dog_bark", 0.80).fired)
    print(f"  → 告警 {fired} 次（冷却 {fsm.policies['dog_bark'].cooldown_frames} 帧抑制刷屏）")

    print("\n[场景 2] 玻璃破碎：2 帧 0.40（低阈值 + 2 连击）")
    fsm2 = AsymmetricFSM()
    seq = [0.1, 0.42, 0.45, 0.1, 0.1]
    for s in seq:
        d = fsm2.step("glass_break", s)
        print(f"  score={s:.2f} → {d.reason:<20} state={d.state_after:<10} fired={d.fired}")
    print(f"  → 告警 {fsm2.states['glass_break'].alarms} 次（2 连击即触发，宁误勿漏）")

    print("\n[场景 3] 单帧尖峰：犬吠 1 帧 0.95 后回落")
    fsm3 = AsymmetricFSM()
    for s in [0.95, 0.2, 0.2]:
        fsm3.step("dog_bark", s)
    print(f"  → 告警 {fsm3.states['dog_bark'].alarms} 次（单帧不触发，连续帧挡误报）")

    print("\n=== 结论 ===")
    print("  同一套打分，两类事件得到完全不同的行为：怕漏检的 2 连击就报，")
    print("  怕误报的要 4 连击 + 90 帧冷却。阈值不是调出来的，是按代价配出来的。")


if __name__ == "__main__":
    demo()
