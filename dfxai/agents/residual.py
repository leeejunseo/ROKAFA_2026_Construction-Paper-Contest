"""
잔차형 혼합(residual hybrid, RES): 규칙이 기본 지령을 내고 학습이 **보정치만** 더한다.

    a_final = a_BT-v2 + β · δ(o),   δ(o) = tanh(MLP(o)) ∈ [-1, 1]^3

선형 혼합(hybrid.py)은 두 정책이 각자 완전한 지령을 낸 뒤 평균을 내므로, 학습 정책이
규칙과 정반대 지령을 내면 결과는 어느 쪽도 아닌 행동이 된다. 잔차형은 학습 정책이
규칙 지령을 **뒤집을 수 없고 다듬기만** 할 수 있다(보정 허용폭 β). 그래서
"규칙의 뼈대는 남기고 학습이 미세조정만 하면 설명 비용을 덜 치르는가"가 이 계열의
질문이다.

채널별 합성:
  * 뱅크(각도, ±1 = ±180°): 더한 뒤 **원형으로 감는다**(clip 하면 ±180° 근처에서 불연속).
  * 하중배수·스로틀: 더한 뒤 [-1, 1] 로 자른다.

초기화: 가중치를 0 근방(표준편차 0.01)에서 시작해 δ ≈ 0, 즉 처음에는 BT-v2 와 거의
같은 행동에서 출발한다(행동 복제 초기화보다 단순하고 같은 효과).
"""
from __future__ import annotations
import numpy as np

from .base import Policy
from .bt import BTPolicy
from .hybrid import MLPPolicy


def _wrap_bank(x: float) -> float:
    """정규화 뱅크(±1 = ±180°)를 원형으로 감는다."""
    return ((x + 1.0) % 2.0) - 1.0


class ResidualPolicy(Policy):
    def __init__(self, beta: float = 0.35, bt_version: int = 2, hidden=(32, 32),
                 params: np.ndarray | None = None, seed: int = 0):
        self.beta = float(beta)
        self.bt_version = int(bt_version)
        self.bt = BTPolicy(version=self.bt_version)
        if params is None:
            rng = np.random.default_rng(seed)
            proto = MLPPolicy(hidden=hidden, seed=seed)
            params = rng.normal(0.0, 0.01, proto.n_params)      # δ ≈ 0 에서 출발
        self.mlp = MLPPolicy(hidden=hidden, params=params)
        self.name = f"RES-b{self.beta:.2f}"
        self.last_node = ""
        self.last_delta = np.zeros(3)

    # ES 학습기가 쓰는 평면 파라미터 인터페이스
    @property
    def flat(self) -> np.ndarray:
        return self.mlp.flat

    @property
    def n_params(self) -> int:
        return self.mlp.n_params

    def set_params(self, flat: np.ndarray) -> None:
        self.mlp.set_params(flat)

    def act(self, obs: np.ndarray) -> np.ndarray:
        a_bt = np.asarray(self.bt.act(obs), dtype=np.float64)
        self.last_node = self.bt.last_node
        delta = np.asarray(self.mlp.act(obs), dtype=np.float64)
        self.last_delta = delta
        a = a_bt + self.beta * delta
        out = np.clip(a, -1.0, 1.0)
        out[0] = _wrap_bank(a[0])
        return out

    def complexity(self) -> dict:
        d = {"beta": self.beta, "nn_params": int(self.mlp.n_params)}
        d.update({f"bt_{k}": v for k, v in self.bt.complexity().items()})
        return d

    def save(self, path: str) -> None:
        np.savez(path, kind=np.array("residual"), flat=self.mlp.flat,
                 hidden=np.array(self.mlp.hidden), beta=np.array(self.beta),
                 bt_version=np.array(self.bt_version))

    @staticmethod
    def load(path: str) -> "ResidualPolicy":
        d = np.load(path)
        assert str(d["kind"]) == "residual", f"{path} 는 잔차형 체크포인트가 아닙니다"
        return ResidualPolicy(beta=float(d["beta"]), bt_version=int(d["bt_version"]),
                              hidden=tuple(int(x) for x in d["hidden"]), params=d["flat"])
