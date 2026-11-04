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
from .bt import _AC
from .registry import make_bt, safety_nodes
from ..dynamics import max_load_factor
from ..geometry import denorm

_G_CAP = 8.0


class ShieldPolicy(Policy):
    """bt: 안전 노드를 가져올 행동트리 이름(registry). 본실험은 BT-v3, C안은 기준선 BT.
    가로채는 노드는 그 행동트리의 SAFETY_NODES(BT-v3 는 합류 회피·하드덱 회복).
    학습 정책의 G 상한은 그 행동트리의 자율 G 제한(g_cap, BT-M 은 7.7 G)이고, 없으면 본실험의 8 G."""

    def __init__(self, rl: Policy, bt_version=3, bt=None):
        self.rl = rl
        self.bt = make_bt(bt if bt is not None else bt_version)
        self._safety = safety_nodes(self.bt)
        self.g_cap = float(getattr(self.bt, "g_cap", _G_CAP))
        self.name = "SHD"
        self.last_node = ""

    def act(self, obs: np.ndarray) -> np.ndarray:
        a_bt = self.bt.act(obs)
        if self.bt.last_node in self._safety:       # 안전 규칙이 발동 → 규칙이 지령
            self.last_node = self.bt.last_node
            return a_bt
        a = np.asarray(self.rl.act(obs), dtype=np.float64).copy()
        # 과G 방패: 현재 가용 G 가 G 상한(self.g_cap)을 넘을 때 지령을 그 상한에 해당하는 값으로 자른다.
        d = denorm(obs)
        n_avail = max_load_factor(d["v_own"], d["h_own"], _AC)
        if n_avail > self.g_cap:
            a[1] = min(a[1], 2.0 * self.g_cap / n_avail - 1.0)
        self.last_node = "rl"
        return np.clip(a, -1.0, 1.0)

    def complexity(self) -> dict:
        d = {"shield_nodes": len(self._safety)}
        d.update({f"rl_{k}": v for k, v in self.rl.complexity().items()})
        return d
