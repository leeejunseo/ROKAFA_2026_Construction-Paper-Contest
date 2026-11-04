"""
BT-M / BT-M+ (손형래, agents/bt_manual.py) 를 학습·평가 도구의 형식(agents/registry.py)에 맞추는 어댑터.

bt_manual.py 는 손대지 않는다. 여기서 덧붙이는 것:
  * SAFETY_NODES 를 클래스 속성으로 (감독형이 가로챌 노드: 통과 이격, 하드덱 회복)
  * rule_count() = 행동(Action) 노드 수
  * PARAM_SPEC 과 __init__(theta) — 파라미터 최적화형용. **상수 선택과 범위는 이준서의 제안이며
    손형래(H3)가 확정한다.** 안전 상수(통과 이격·하드덱·자율 G 제한)는 최적화하지 않는다:
    최적화된 BT 도 기준선과 같은 안전 규칙을 지켜야 "규칙은 그대로, 상수만 조정"이 되기 때문이다.

registry 이름: btm = BT-M (기준선), btmplus = BT-M+ (상세판)
"""
from __future__ import annotations
import math
from dataclasses import replace

import numpy as np

from .bt import Action
from .bt_manual import ManualBTPolicy, BTMParams, SAFETY_NODES as _SAFETY

_D = math.radians

# (BTMParams 필드, 하한, 상한, 기본값) — 각도는 라디안, 거리 m, 속도 m/s. θ = -1 → 하한, +1 → 상한 (제안)
_DEF = BTMParams()
PARAM_SPEC = [
    ("def_threat_ata", _D(25.0), _D(70.0), _DEF.def_threat_ata),   # 방어 전환: 상대 기수가 나를 겨누는 각
    ("def_range",      1500.0,   4000.0,   _DEF.def_range),        # 방어 전환 거리
    ("tc_range",       1000.0,   3000.0,   _DEF.tc_range),         # 선회원 크기 (고 G 당김 시작 거리)
    ("entry_g",        2.5,      7.0,      _DEF.entry_g),          # 선회원 밖 접근 하중 [G]
    ("gun_range",      800.0,    2000.0,   _DEF.gun_range),        # 리드추격 전환 거리
    ("gun_ata",        _D(20.0), _D(60.0), _DEF.gun_ata),          # 리드추격 전환 총각
    ("lead_gain",      0.8,      1.8,      _DEF.lead_gain),        # 리드 이득
    ("v_rate_hi",      190.0,    260.0,    _DEF.v_rate_hi),        # 코너 플래토 상한 (넘으면 감속)
]


def _theta_to_params(base: BTMParams, theta) -> BTMParams:
    kw = {}
    for (name, lo, hi, _), t in zip(PARAM_SPEC, np.asarray(theta, float)):
        t = min(1.0, max(-1.0, float(t)))
        kw[name] = lo + (hi - lo) * 0.5 * (t + 1.0)
    return replace(base, **kw)


class BTMPolicy(ManualBTPolicy):
    """registry 'btm'. theta 가 없으면 bt_manual 의 BT-M 과 완전히 같은 정책."""
    SAFETY_NODES = _SAFETY
    PARAM_SPEC = PARAM_SPEC
    _base = staticmethod(BTMParams.core)

    def __init__(self, theta=None):
        P = self._base()
        if theta is not None:
            P = _theta_to_params(P, theta)
        super().__init__(P)
        self.theta = None if theta is None else np.asarray(theta, float)

    def rule_count(self) -> int:
        def walk(n):
            return int(isinstance(n, Action)) + sum(walk(c) for c in n.children)
        return walk(self.root)


class BTMPlusPolicy(BTMPolicy):
    """registry 'btmplus' (BT-M+, 상세 규칙 켬)."""
    _base = staticmethod(BTMParams.full)
