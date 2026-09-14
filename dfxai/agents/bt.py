"""
행동트리(Behavior Tree) 기반 규칙 모델.

설계 원칙 두 가지 — 논문의 '주관성' 방어에 직결됩니다.

1) 전술을 창작하지 않고 **기본전투기동(BFM) 교범의 표준 전술**만 옮깁니다.
   순수추격 / 리드추격 / 래그추격, 코너속도 관리, 에너지 보존,
   브레이크 턴, 하드덱 이탈 회복. 전부 문헌에 이름이 있는 기동이므로
   "박OO이 임의로 만든 BT"라는 지적을 피할 수 있습니다.

2) 한 번에 완성하지 않고 **v1 -> v2 -> v3 누적 버전**으로 만듭니다.
   그래야 BT도 RL과 마찬가지로 '개발 노력에 따른 궤적'을 그릴 수 있고,
   성능-설명가능성 평면에 BT 곡선과 RL 곡선을 동시에 그릴 수 있습니다.
   끝점 두 개를 비교하는 것보다 훨씬 강한 주장이 됩니다.

트리는 실제 Selector/Sequence 노드로 구성하여 노드 수를 셀 수 있게 했습니다.
"""
from __future__ import annotations
import math
import numpy as np
from typing import Callable, List

from .base import Policy
from ..geometry import denorm
from ..config import AircraftConfig
from ..dynamics import max_load_factor

SUCCESS, FAILURE = True, False

_DEG3 = math.radians(3.0)


def _clip(x, lo=-1.0, hi=1.0):
    return lo if x < lo else (hi if x > hi else x)


# ---------------------------------------------------------------- 트리 노드
class Node:
    def __init__(self, name: str):
        self.name = name
        self.children: List["Node"] = []

    def tick(self, bb: dict) -> bool:
        raise NotImplementedError

    def count(self) -> int:
        return 1 + sum(c.count() for c in self.children)

    def describe(self, indent: int = 0) -> str:
        s = "  " * indent + f"- {self.__class__.__name__}: {self.name}\n"
        for c in self.children:
            s += c.describe(indent + 1)
        return s


class Selector(Node):
    """자식을 순서대로 시도, 처음 성공하는 자식에서 멈춤 (우선순위 분기)."""

    def __init__(self, name, children):
        super().__init__(name)
        self.children = children

    def tick(self, bb):
        for c in self.children:
            if c.tick(bb):
                return SUCCESS
        return FAILURE


class Sequence(Node):
    """자식이 모두 성공해야 성공 (조건 AND 행동)."""

    def __init__(self, name, children):
        super().__init__(name)
        self.children = children

    def tick(self, bb):
        for c in self.children:
            if not c.tick(bb):
                return FAILURE
        return SUCCESS


class Condition(Node):
    def __init__(self, name, fn: Callable[[dict], bool]):
        super().__init__(name)
        self.fn = fn

    def tick(self, bb):
        return bool(self.fn(bb))


class Action(Node):
    def __init__(self, name, fn: Callable[[dict], None]):
        super().__init__(name)
        self.fn = fn

    def tick(self, bb):
        self.fn(bb)
        bb["fired_node"] = self.name
        return SUCCESS


# ---------------------------------------------------------------- 보조 함수
_AC = AircraftConfig()


def _corner_speed(h: float) -> float:
    """해당 고도에서 구조한계 G를 낼 수 있는 최소 속도 (근사 코너속도)."""
    from ..dynamics import air_density
    from ..config import G0
    rho = air_density(h)
    q_need = _AC.n_max_struct * _AC.mass * G0 / (_AC.wing_area * _AC.cl_max)
    return math.sqrt(2.0 * q_need / rho)


def _point_at(bb: dict, lead_gain: float = 1.0, g_frac: float = 1.0,
              throttle: float = 1.0) -> None:
    """3차원 순수추격: 양력벡터를 시선 방향으로 돌립니다.

    수평 이탈각(ata_h)과 수직 이탈각(ata_v)을 합성해 필요한 뱅크각을 구합니다.
    lead_gain > 1 이면 리드추격(사격 전 예측조준), < 1 이면 래그추격입니다.
    g_frac 는 '현재 가용 최대 G 중 몇 퍼센트를 쓸 것인가'입니다.
    """
    ah = bb["ata_h"] * lead_gain
    av = bb["ata_v"]
    if abs(ah) < 1e-3 and abs(av) < 1e-3:
        mu_des = 0.0                      # 이미 정조준: 날개 수평
    else:
        # av < 0 (목표가 아래)이면 |mu| > 90deg 가 되어 기수 하향 선회가 됩니다.
        mu_des = math.atan2(ah, av)
    bb["a"][0] = _clip(mu_des / _AC.bank_cmd_limit)
    bb["a"][1] = _clip(2.0 * g_frac - 1.0)
    bb["a"][2] = _clip(2.0 * throttle - 1.0)


def _horizontal_pursuit(bb: dict, bank_deg: float = 75.0, g: float = 4.0,
                        throttle: float = 0.8) -> None:
    """수평면만 사용하는 조잡한 추격. 수직 이탈각을 무시합니다."""
    ah = bb["ata_h"]
    mu = math.copysign(math.radians(bank_deg), ah) if abs(ah) > _DEG3 else 0.0
    bb["a"][0] = _clip(mu / _AC.bank_cmd_limit)
    bb["a"][1] = _clip(2.0 * (g / _AC.n_max_struct) - 1.0)
    bb["a"][2] = _clip(2.0 * throttle - 1.0)


def _hold_level(bb: dict, g: float = 4.0, throttle: float = 0.8) -> None:
    """기수가 이미 정렬된 경우 날개 수평 유지."""
    bb["a"][0] = 0.0
    bb["a"][1] = _clip(2.0 * (g / _AC.n_max_struct) - 1.0)
    bb["a"][2] = _clip(2.0 * throttle - 1.0)


_MERGE_R = 1000.0                 # [m] 이 거리 안에서
_MERGE_CLOSURE = 200.0            # [m/s] 이보다 빠르게 접근 중이고
_MERGE_AA = np.deg2rad(140.0)     # 상대도 나를 향해 있고 (AA 가 정면에 가까움)
_MERGE_ATA = np.deg2rad(40.0)     # 나도 상대를 향해 있으면 -> 정면 합류
_MERGE_BANK = np.deg2rad(75.0)
_MERGE_G = 5.0


def _head_on_merge(bb: dict) -> bool:
    """정면 합류(head-on merge) 판정.

    순수추격만 있는 BT 는 정면 조우에서 서로를 향해 직진하다 공중충돌합니다
    (사전실험에서 교전의 45~53%). 실제 BFM 에서는 합류 시 측방 이격을
    유지하는 것이 기본 규칙이므로, 세 BT 모두에 이 회피 노드를 넣습니다.
    """
    return (bb["r"] < _MERGE_R and bb["closure"] > _MERGE_CLOSURE
            and bb["aa"] > _MERGE_AA and bb["ata_total"] < _MERGE_ATA)


def _merge_offset(bb: dict) -> None:
    """합류 회피: 상대가 있는 쪽의 반대로 기울여 측방 이격을 만듭니다.

    양측이 같은 규칙을 쓰면 각자 '자기 기준 반대편'으로 꺾으므로 세계 좌표에서는
    서로 멀어지는 방향이 됩니다. 정확히 정면(ata_h = 0)이면 우측으로 꺾습니다
    (항공 규칙의 우측 회피와 같은 관례).
    """
    ah = bb["ata_h"]
    side = -1.0 if ah > 1e-3 else 1.0
    bb["a"][0] = _clip(side * _MERGE_BANK / _AC.bank_cmd_limit)
    bb["a"][1] = _clip(2.0 * (_MERGE_G / _AC.n_max_struct) - 1.0)
    bb["a"][2] = 1.0


def _merge_node() -> Node:
    return Sequence("merge_avoidance", [
        Condition("head_on_merge", _head_on_merge),
        Action("merge_offset", _merge_offset),
    ])


def _level_climb(bb: dict) -> None:
    """날개를 수평으로 되돌리고 완만히 상승. 하드덱 이탈 회복."""
    bb["a"][0] = 0.0
    bb["a"][1] = 2.0 * 0.35 - 1.0
    bb["a"][2] = 1.0


# ---------------------------------------------------------------- 트리 정의
#
# 세 버전은 '개발 성숙도 사다리'가 아니라 **서로 다른 전술 교리**입니다.
# 셋 다 정면 합류 회피 노드(_merge_node)를 최우선으로 공유합니다.
# 사전실험(100시드 진영교대) 성능 서열은 v2 > v3 > v1 이고,
# 규칙 준수 서열은 v3 > v1 > v2 로 정반대입니다.
# 즉 이 BT 3종만으로도 '성능 대비 규칙 준수' 트레이드오프가 이미 관측됩니다.
# 성능 서열을 가정하지 말고 반드시 측정해서 보고하세요.

def build_tree_v1() -> Node:
    """BT-A '수평 추격' — 수직면을 쓰지 않고 고정 4G로만 선회."""
    return Selector("root_v1_horizontal", [
        _merge_node(),
        Sequence("nose_on_hold", [
            Condition("bearing_aligned",
                      lambda bb: abs(bb["ata_h"]) <= _DEG3),
            Action("hold_wings_level_fixed_g", _hold_level),
        ]),
        Action("horizontal_pursuit_fixed_g", _horizontal_pursuit),
    ])


def build_tree_v2() -> Node:
    """BT-B '3차원 최대선회' — 양력벡터 지향 + 최대 G + 최대 추력.

    성능은 가장 높지만 하드덱과 과G 한계를 상시 침범합니다.
    """
    return Selector("root_v2_max_rate", [
        _merge_node(),
        Action("pursuit_3d_max_g", lambda bb: _point_at(bb, 1.0, 1.0, 1.0)),
    ])


def build_tree_v3() -> Node:
    """BT-C '3차원 + 안전규칙' — v2에 하드덱 회복, 과G 억제, 리드추격 추가.

    성능을 일부 내주는 대신 규칙 위반을 0으로 만듭니다.
    """
    g_cap = 8.0 / _AC.n_max_struct        # 구조한계 9G 대신 8G로 자율 제한
    return Selector("root_v3_compliant", [
        _merge_node(),
        Sequence("deck_recovery", [
            Condition("below_deck_margin",
                      lambda bb: bb["h_own"] < bb["deck"] + 400.0
                      and bb["gamma_own"] < 0.05),
            Action("level_climb", _level_climb),
        ]),
        Sequence("gun_tracking", [
            Condition("in_gun_envelope",
                      lambda bb: bb["r"] < 1100.0 and bb["ata_total"] < 0.20),
            Action("lead_pursuit_limited_g",
                   lambda bb: _point_at(bb, 1.25, g_cap, 1.0)),
        ]),
        Action("pursuit_3d_limited_g", lambda bb: _point_at(bb, 1.0, g_cap, 1.0)),
    ])


_BUILDERS = {1: build_tree_v1, 2: build_tree_v2, 3: build_tree_v3}


class BTPolicy(Policy):
    """행동트리 정책. version 으로 개발 성숙도 단계를 선택합니다."""

    def __init__(self, version: int = 3, hard_deck: float = 1000.0):
        assert version in _BUILDERS, f"version must be one of {list(_BUILDERS)}"
        self.version = version
        self.root = _BUILDERS[version]()
        self.hard_deck = hard_deck
        self.name = f"BT-v{version}"
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
        return {"bt_nodes": self.root.count(), "bt_version": self.version}

    def describe(self) -> str:
        return self.root.describe()
