"""
교범 기반 평가 상대 BT-R: 반경 싸움형 (C안, 손형래).

BT-M 이 코너 플래토(330~440 KCAS)에서 선회율로 싸운다면, BT-R 은 같은 교범이 말하는
"반경 싸움"을 한다. 330 KCAS 를 넘으면 추력을 줄여 빨리 감속하고, 330 KCAS 아래에서는 후기연소로
최대 G 를 유지하며 가장 작은 선회 반경으로 기수를 먼저 돌린다.

    * 330 KCAS 위: 추력을 줄여 반경을 빨리 줄인다. 아래: 후기연소 (4.6.36.1)
    * 130~329 KCAS 가 가장 작은 선회 반경 구간 (4.6.12 Rate versus Radius)
    * 1서클 싸움은 반경 싸움이며 기수를 먼저 돌리는 쪽이 이긴다 (4.6.9.2, 4.6.13)
    * 정면에 가까운 각도에서도 사격 기회를 잡는다 (4.4.15 High-Aspect Gunshot)

안전 하위트리, 바닥 구역 하강 선회 제한, 7.7 G 상한, 관측·지령 형식은 BT-M 과 같다.
평가 전용 상대이며 학습에는 쓰지 않는다.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, asdict

import numpy as np

from .base import Policy
from .bt import Node, Selector, Sequence, Condition, Action
from .bt_manual import (ManualBTPolicy, BTMParams, SAFETY_NODES as _SAFETY,
                        _set, _pursuit_bank, _clip, FT, KT, _D)
from ..config import EngagementConfig


@dataclass
class BTRParams:
    """BT-R 조정 상수. 주석 괄호 안은 교범 원값과 절 번호."""
    v_radius: float = 330.0 * KT          # 이 위면 추력 감소, 아래면 후기연소 (4.6.36.1)
    thr_slow: float = 0.0                 # 감속할 때 스로틀 ("reduce power to minimize the radius as quickly as possible")
    gun_range: float = 4500.0 * FT        # 사격 거리 [m] (4.4.16 3,500~4,500 ft)
    gun_ata: float = _D(40.0)             # 기수 40° 안이면 애스펙트와 상관없이 리드 사격 (4.4.15)
    lead_gain: float = 1.3
    def_threat_ata: float = _D(45.0)
    def_my_ata: float = _D(90.0)
    def_range: float = 9000.0 * FT
    cpa_time: float = 2.5                 # 통과 회피 판단 시간 [s] (BT-M 과 같은 안전 하위트리의 상수)

    def as_dict(self) -> dict:
        return {k: (round(math.degrees(v), 3) if k.endswith("_ata") else round(v, 3))
                for k, v in asdict(self).items()}


class RadiusBTPolicy(Policy):
    """교범 기반 반경 싸움형 행동트리 BT-R (평가 전용 상대)."""

    SAFETY_NODES = _SAFETY

    def __init__(self, params: BTRParams | None = None, ec: EngagementConfig | None = None):
        self.R = params or BTRParams()
        self._m = ManualBTPolicy(BTMParams.core(cpa_time=self.R.cpa_time), ec=ec)   # 안전 하위트리·관측 해석은 BT-M 과 공유
        self.P = self._m.P
        self.g_cap = self._m.g_cap
        self.root = self._build()
        self.name = "BT-R"
        self.last_node = ""

    def _build(self) -> Node:
        R, P, g_cap = self.R, self.P, self.g_cap

        def radius_power(bb):
            return R.thr_slow if bb["v_eas"] > R.v_radius else 1.0

        def gun_window(bb):
            return bb["r"] < R.gun_range and bb["ata_total"] < R.gun_ata

        def high_aspect_lead(bb):
            _set(bb, _pursuit_bank(bb, P, R.lead_gain), g_cap, radius_power(bb))

        def threatened(bb):
            return (bb["thr_ata"] < R.def_threat_ata and bb["ata_total"] > R.def_my_ata
                    and bb["r"] < R.def_range)

        def break_turn(bb):
            _set(bb, _pursuit_bank(bb, P, 1.0), g_cap, radius_power(bb))

        def radius_turn(bb):
            _set(bb, _pursuit_bank(bb, P, 1.0), g_cap, radius_power(bb))

        return Selector("root_BT-R", [
            self._m.safety,
            Sequence("gun_any_aspect", [Condition("gun_window", gun_window),
                                        Action("high_aspect_lead", high_aspect_lead)]),
            Sequence("DEFENSIVE", [Condition("threatened", threatened),
                                   Action("break_turn_into_threat", break_turn)]),
            Action("min_radius_turn", radius_turn),
        ])

    def act(self, obs: np.ndarray) -> np.ndarray:
        bb = self._m._blackboard(obs)
        self.root.tick(bb)
        self.last_node = bb["fired_node"]
        a = bb["a"]
        return np.array([_clip(a[0]), _clip(a[1]), _clip(a[2])])

    def safety_act(self, obs: np.ndarray):
        return self._m.safety_act(obs)

    def complexity(self) -> dict:
        return {"bt_nodes": self.root.count(), "bt_version": self.name, "g_cap": self.g_cap}

    def describe(self) -> str:
        return self.root.describe()
