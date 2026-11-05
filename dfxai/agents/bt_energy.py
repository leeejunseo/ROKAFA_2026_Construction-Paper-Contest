"""
교범 기반 학습 상대 BT-E: 에너지·수직 전투형 (C안, 손형래).

BT-M 이 "선회율 싸움형"(상대 쪽으로 최대 G 로 계속 당기는 2서클 선회율 싸움)이라면,
BT-E 는 같은 교범의 다른 게임플랜을 옮긴 "에너지 전투형"이다.

    * 에너지를 아낀다: 사격 기회가 아니면 최대 G 대신 지속 선회 하중(약 5.5 G)으로만 당기고
      400 KCAS 아래에서는 추력을 최대로 유지한다 (4.6.36.2 second technique).
    * 속도가 330 KCAS 아래로 떨어지면 하중을 풀고 가속한다 (4.3.3.18, 4.6.12).
    * 수직을 쓴다: 고도 여유가 있으면 양력벡터를 상대 리드 쪽 아래로 두는 기수 하향
      면외 기동으로 애스펙트 우위를 만든다 (4.6.31 Using Vertical to Create an Advantageous Pass).
    * 방어 시 거리가 있으면 브레이크 대신 하중을 풀고 가속 이탈한다 (4.3.3.18 acceleration/separation,
      4.4.8 break and extend). 가까우면 BT-M 처럼 브레이크 턴.
    * 사격 기회(거리·각도)가 오면 그때만 최대 G 리드 추격 (4.4.14~4.4.16).

안전 하위트리(통과 이격, 하드덱 회복), 바닥 구역의 하강 선회 제한, 7.7 G 상한, 관측·지령 형식은
BT-M 과 완전히 같다. bt_manual.py 는 고치지 않고 그 안전 하위트리와 보조 함수를 가져다 쓴다.
교범 근거와 값 변환은 docs/BT-M_교범대응표.md 의 BT-E 절.
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
class BTEParams:
    """BT-E 조정 상수. 주석 괄호 안은 교범 원값과 절 번호."""
    # 사격 기회 (4.4.16)
    gun_range: float = 4500.0 * FT        # 리드 추격 전환 거리 [m] (3,500~4,500 ft)
    gun_ata: float = _D(40.0)             # 상대가 기수 40° 안
    lead_gain: float = 1.3
    # 방어 (4.5, 4.3.3.18)
    def_threat_ata: float = _D(45.0)
    def_my_ata: float = _D(90.0)
    def_range: float = 9000.0 * FT        # 9,000 ft
    break_range: float = 6000.0 * FT      # 이 안이거나 상대가 접근 중이면 브레이크 턴 (선회원 6,000 ft)
    #                                       밖이면서 이미 멀어지는 중일 때만 가속 이탈 (4.3.3.18: 이탈은 상대가
    #                                       쫓지 않을 때만 성공한다)
    extend_g: float = 0.8                 # 이탈 하중 [G] (4.3.3.17 "0.7 G to 0.9 G slightly nose-low extension")
    # 에너지 관리 (4.6.36.2, 4.6.12)
    v_min: float = 330.0 * KT             # 이 아래면서 선회원 밖이면 하중을 풀고 가속 (코너 플래토 하한)
    v_power: float = 440.0 * KT           # 이 위면 추력 감소 (코너 플래토 상한)
    thr_fast: float = 0.30
    accel_g: float = 1.5                  # 가속 중 하중 [G] (상대 쪽으로 완만히)
    sustain_g: float = 6.4                # 에너지 보존 선회 하중 [G]: 표 4.9 의 정상 선회(0 Ps) 425 KIAS 13.5°/s 에 해당
    #                                       (같은 표 425 KCAS 행: 6 G +3 kt/s, 7 G -5 kt/s 의 사이)
    fight_range: float = 6000.0 * FT      # 선회원(6,000 ft) 안에서는 최대 G 로 싸운다. 에너지 보존은 그 밖에서만
    # 수직 기동 (4.6.31, 4.6.32, 4.4.10)
    vert_alt: float = 5000.0 * FT         # 하드덱 위 이 고도 이상에서만 (4,000 ft 규칙 + 여유)
    vert_rmin: float = 3000.0 * FT        # 상대와 한 선회 반경(약 3,000 ft) 이상 떨어져 있을 때
    vert_bias: float = _D(-10.0)          # 양력벡터를 상대보다 아래로 (기수 하향 면외 기동). 교범에 각도 수치가 없어
    #                                       시험(각 60교전)으로 정함: -25° 는 BT-M 상대 점수 0.13~0.31 로 너무 약했다
    vert_g: float = -1.0                  # < 0 이면 자율 G 상한(7.7 G): 중력 방향으로 도는 수직 선회는 최대 G
    #                                       (4.3.3.14 "maximum G ... best employed when working with gravity")

    def as_dict(self) -> dict:
        ang = ("_ata", "_bias")
        return {k: (round(math.degrees(v), 3) if k.endswith(ang) else round(v, 3))
                for k, v in asdict(self).items()}


class EnergyBTPolicy(Policy):
    """교범 기반 에너지·수직 전투형 행동트리 BT-E."""

    SAFETY_NODES = _SAFETY

    def __init__(self, params: BTEParams | None = None, ec: EngagementConfig | None = None):
        self.E = params or BTEParams()
        self._m = ManualBTPolicy(BTMParams.core(), ec=ec)   # 안전 하위트리·관측 해석은 BT-M 과 공유
        self.P = self._m.P
        self.g_cap = self._m.g_cap
        self.root = self._build()
        self.name = "BT-E"
        self.last_node = ""

    # ------------------------------------------------------------------ 트리
    def _build(self) -> Node:
        E, P, g_cap = self.E, self.P, self.g_cap

        def power(bb):                                     # 400 KCAS 아래에서는 추력 유지 (4.6.36.2)
            return E.thr_fast if bb["v_eas"] > E.v_power else 1.0

        # 사격 기회
        def gun_window(bb):
            return bb["r"] < E.gun_range and bb["ata_total"] < E.gun_ata

        def snapshot_lead(bb):
            _set(bb, _pursuit_bank(bb, P, E.lead_gain), g_cap, 1.0)

        # 방어
        def threatened(bb):
            return (bb["thr_ata"] < E.def_threat_ata and bb["ata_total"] > E.def_my_ata
                    and bb["r"] < E.def_range)

        def threat_close(bb):
            return bb["r"] < E.break_range or bb["closure"] > 0.0

        def break_turn(bb):
            _set(bb, _pursuit_bank(bb, P, 1.0), g_cap, power(bb))

        def unload_extend(bb):
            g = E.extend_g if bb["h_deck"] > P.floor_zone else 1.0   # 바닥 근처에서는 수평 유지
            _set(bb, 0.0, g, 1.0)

        # 에너지
        def energy_low(bb):
            # 가속은 교전 거리 밖에서만: 선회원 안에서 하중을 풀면 상대에게 사격 기회를 준다
            return bb["v_eas"] < E.v_min and bb["r"] > E.fight_range

        def unload_accelerate(bb):
            _set(bb, _pursuit_bank(bb, P, 1.0), E.accel_g, 1.0)

        # 수직
        def vertical_window(bb):
            return (bb["h_deck"] > E.vert_alt and bb["r"] > E.vert_rmin
                    and bb["ata_total"] < _D(90.0) and bb["aa"] > _D(60.0))

        def nose_low_lead(bb):
            g = g_cap if E.vert_g < 0 else E.vert_g
            _set(bb, _pursuit_bank(bb, P, E.lead_gain, E.vert_bias), g, 1.0)

        # 기본
        def sustained_pursuit(bb):
            g = g_cap if bb["r"] < E.fight_range else E.sustain_g
            _set(bb, _pursuit_bank(bb, P, 1.0), g, power(bb))

        return Selector("root_BT-E", [
            self._m.safety,
            Sequence("gun_opportunity", [Condition("gun_window", gun_window),
                                         Action("snapshot_lead", snapshot_lead)]),
            Sequence("DEFENSIVE", [
                Condition("threatened", threatened),
                Selector("defensive_select", [
                    Sequence("break_if_close", [Condition("threat_close", threat_close),
                                                Action("break_turn_into_threat", break_turn)]),
                    Action("unload_extend", unload_extend),
                ]),
            ]),
            Sequence("energy_rebuild", [Condition("energy_low", energy_low),
                                        Action("unload_accelerate", unload_accelerate)]),
            Sequence("vertical_attack", [Condition("vertical_window", vertical_window),
                                         Action("nose_low_lead", nose_low_lead)]),
            Action("sustained_pursuit", sustained_pursuit),
        ])

    # ------------------------------------------------------------------ 정책
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
