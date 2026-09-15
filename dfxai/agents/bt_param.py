"""
파라미터화 행동트리(BTO, "optimized BT").

왜 필요한가 — 박사 수준 심사의 첫 반론
--------------------------------------
본실험의 BT 3종은 손으로 정한 상수(G 한계, 사거리 문턱, 리드 게인 …)를 쓴다.
그래서 "규칙 모델이 약하다"는 결과는 "우리가 손으로 만든 규칙 모델이 약하다"일
뿐일 수 있다(Rudin 2019: 해석 가능한 모델도 최적화하면 강하다). 이 반론을
막으려면 **구조는 BT 그대로 두고 상수만 학습 정책과 같은 진화전략·같은 예산으로
최적화한** 네 번째 계열이 필요하다. 이 모듈이 그것이다.

구조는 BT-v3 와 동일(합류 회피 → 하드덱 회복 → 사격 추적 → 3차원 추격)하고,
아래 9개 상수를 θ ∈ [-1, 1]^9 로 두어 ES 가 움직인다. 노드 수(11)와 트리 구조는
고정이므로 "사람이 읽을 수 있는 규칙"이라는 성질은 유지된다.
"""
from __future__ import annotations
import math
import numpy as np

from .base import Policy
from .bt import (Selector, Sequence, Condition, Action, Node, _point_at,
                 _corner_speed, _clip, _AC)
from ..geometry import denorm

# (이름, 하한, 상한, BT-v3 기본값)  — θ=-1 → 하한, θ=+1 → 상한
PARAM_SPEC = [
    ("g_frac",        0.45, 1.00, 8.0 / 9.0),   # 추격 시 가용 G 중 사용 비율
    ("lead_gain",     0.70, 1.60, 1.25),        # 사격 추적 시 리드 게인
    ("gun_r",         600., 1500., 1100.),      # 사격 추적 진입 거리 [m]
    ("gun_ata",       0.08, 0.50, 0.20),        # 사격 추적 진입 총각 [rad]
    ("deck_margin",   100., 800., 400.),        # 하드덱 회복 개시 여유 [m]
    ("merge_r",       500., 1500., 1000.),      # 합류 회피 개시 거리 [m]
    ("merge_closure", 100., 300., 200.),        # 합류 회피 접근률 문턱 [m/s]
    ("merge_bank",    40., 90., 75.),           # 합류 회피 뱅크 [deg]
    ("throttle",      0.60, 1.00, 1.00),        # 추격 시 스로틀
]
N_PARAMS = len(PARAM_SPEC)

_MERGE_AA = math.radians(140.0)
_MERGE_ATA = math.radians(40.0)


def theta_to_params(theta: np.ndarray) -> dict:
    out = {}
    for (name, lo, hi, _), t in zip(PARAM_SPEC, theta):
        t = _clip(float(t), -1.0, 1.0)
        out[name] = lo + (hi - lo) * 0.5 * (t + 1.0)
    return out


def default_theta() -> np.ndarray:
    """BT-v3 상수에 해당하는 θ (학습 출발점)."""
    th = []
    for name, lo, hi, d in PARAM_SPEC:
        th.append(2.0 * (d - lo) / (hi - lo) - 1.0)
    return np.asarray(th, dtype=np.float64)


def build_param_tree(p: dict) -> Node:
    bank = math.radians(p["merge_bank"])
    g_cap = p["g_frac"]

    def head_on(bb):
        return (bb["r"] < p["merge_r"] and bb["closure"] > p["merge_closure"]
                and bb["aa"] > _MERGE_AA and bb["ata_total"] < _MERGE_ATA)

    def merge_offset(bb):
        side = -1.0 if bb["ata_h"] > 1e-3 else 1.0
        bb["a"][0] = _clip(side * bank / _AC.bank_cmd_limit)
        bb["a"][1] = _clip(2.0 * (5.0 / _AC.n_max_struct) - 1.0)
        bb["a"][2] = 1.0

    def level_climb(bb):
        bb["a"][0] = 0.0
        bb["a"][1] = 2.0 * 0.35 - 1.0
        bb["a"][2] = 1.0

    return Selector("root_bto", [
        Sequence("merge_avoidance", [Condition("head_on_merge", head_on),
                                     Action("merge_offset", merge_offset)]),
        Sequence("deck_recovery", [
            Condition("below_deck_margin",
                      lambda bb: bb["h_own"] < bb["deck"] + p["deck_margin"]
                      and bb["gamma_own"] < 0.05),
            Action("level_climb", level_climb)]),
        Sequence("gun_tracking", [
            Condition("in_gun_envelope",
                      lambda bb: bb["r"] < p["gun_r"] and bb["ata_total"] < p["gun_ata"]),
            Action("lead_pursuit_limited_g",
                   lambda bb: _point_at(bb, p["lead_gain"], g_cap, p["throttle"]))]),
        Action("pursuit_3d_limited_g",
               lambda bb: _point_at(bb, 1.0, g_cap, p["throttle"])),
    ])


class ParamBTPolicy(Policy):
    """상수가 학습된 행동트리. 구조·노드 수는 BT-v3 와 같다."""

    def __init__(self, theta: np.ndarray | None = None, hard_deck: float = 1000.0):
        self.theta = default_theta() if theta is None else np.asarray(theta, float)
        self.params = theta_to_params(self.theta)
        self.root = build_param_tree(self.params)
        self.hard_deck = hard_deck
        self.name = "BTO"
        self.last_node = ""

    def act(self, obs: np.ndarray) -> np.ndarray:
        bb = denorm(obs)
        bb["a"] = [0.0, 0.0, 0.0]
        bb["deck"] = self.hard_deck
        bb["v_corner"] = _corner_speed(bb["h_own"])
        bb["fired_node"] = ""
        self.root.tick(bb)
        self.last_node = bb["fired_node"]
        a = bb["a"]
        return np.array([_clip(a[0]), _clip(a[1]), _clip(a[2])])

    def complexity(self) -> dict:
        return {"bt_nodes": self.root.count(), "n_tuned_params": N_PARAMS}

    def describe(self) -> str:
        lines = [f"- {k}: {v:.3f}" for k, v in self.params.items()]
        return self.root.describe() + "\n".join(lines) + "\n"

    def save(self, path: str) -> None:
        np.savez(path, theta=self.theta, kind=np.array("bto"))

    @staticmethod
    def load(path: str) -> "ParamBTPolicy":
        d = np.load(path)
        return ParamBTPolicy(theta=d["theta"])
