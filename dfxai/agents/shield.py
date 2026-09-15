"""
감독형 혼합(shield hybrid, SHD): 규칙이 학습 정책을 **감독**하는 혼합.

본실험의 혼합(hybrid.py)은 두 정책의 지령을 선형으로 섞는다. 그 결과가
"어느 쪽보다 설명하기 어렵다"는 것은 예상 가능한 결과라 심사에서 허수아비
비교라는 지적을 받는다. 이 모듈은 문헌에서 실제로 쓰는 형태를 구현한다.

    안전 조건이 걸리면 → BT-v3 의 안전 노드(합류 회피·하드덱 회복)가 지령을 낸다
    그 밖에는          → 학습 정책의 지령을 쓰되, 하중배수를 8 G 이하로 자른다

즉 학습 정책은 '무엇을 할지'를 정하고, 규칙은 '해서는 안 되는 것'만 막는다
(safety shield / runtime assurance). 규칙 위반 0 을 학습 정책 위에 얹을 수 있는지,
그때 성능과 설명 비용이 어떻게 되는지가 이 계열의 질문이다.
"""
from __future__ import annotations
import numpy as np

from .base import Policy
from .bt import BTPolicy, _AC
from ..dynamics import max_load_factor
from ..geometry import denorm

_SAFETY_NODES = ("merge_offset", "level_climb")
_G_CAP = 8.0


class ShieldPolicy(Policy):
    def __init__(self, rl: Policy, bt_version: int = 3):
        self.rl = rl
        self.bt = BTPolicy(version=bt_version)
        self.name = "SHD"
        self.last_node = ""

    def act(self, obs: np.ndarray) -> np.ndarray:
        a_bt = self.bt.act(obs)
        if self.bt.last_node in _SAFETY_NODES:      # 안전 규칙이 발동 → 규칙이 지령
            self.last_node = self.bt.last_node
            return a_bt
        a = np.asarray(self.rl.act(obs), dtype=np.float64).copy()
        # 과G 방패: 현재 가용 G 가 8 G 를 넘을 때 지령을 8 G 에 해당하는 값으로 자른다.
        d = denorm(obs)
        n_avail = max_load_factor(d["v_own"], d["h_own"], _AC)
        if n_avail > _G_CAP:
            a[1] = min(a[1], 2.0 * _G_CAP / n_avail - 1.0)
        self.last_node = "rl"
        return np.clip(a, -1.0, 1.0)

    def complexity(self) -> dict:
        d = {"shield_nodes": 2}
        d.update({f"rl_{k}": v for k, v in self.rl.complexity().items()})
        return d
