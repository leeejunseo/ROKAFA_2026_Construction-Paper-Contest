"""
교범 기반 상대 BT-E(에너지·수직형)·BT-R(반경 싸움형)을 registry 형식에 맞추는 어댑터 (손형래).

bt_energy.py·bt_radius.py 는 손대지 않고 여기서 형식만 덧붙인다.
  * 두 BT 는 학습·평가 상대 전용이라 파라미터 최적화형(theta)을 지원하지 않는다.
  * SAFETY_NODES 는 BT-M 과 같다(안전 하위트리를 공유).
registry 이름: bte = BT-E (학습 상대), btr = BT-R (평가 전용 상대)
"""
from __future__ import annotations

from .bt import Action
from .bt_energy import EnergyBTPolicy
from .bt_radius import RadiusBTPolicy


def _rule_count(root) -> int:
    def walk(n):
        return int(isinstance(n, Action)) + sum(walk(c) for c in n.children)
    return walk(root)


class BTEPolicy(EnergyBTPolicy):
    """registry 'bte'."""
    def __init__(self, theta=None):
        if theta is not None:
            raise ValueError("BT-E 는 상대 전용이라 조정 상수(theta)를 받지 않습니다")
        super().__init__()

    def rule_count(self) -> int:
        return _rule_count(self.root)


class BTRPolicy(RadiusBTPolicy):
    """registry 'btr'."""
    def __init__(self, theta=None):
        if theta is not None:
            raise ValueError("BT-R 은 평가 전용 상대라 조정 상수(theta)를 받지 않습니다")
        super().__init__()

    def rule_count(self) -> int:
        return _rule_count(self.root)
