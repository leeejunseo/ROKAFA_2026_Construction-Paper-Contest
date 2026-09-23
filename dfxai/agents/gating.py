"""
게이팅형 혼합(gated hybrid, GATE): 혼합 비율을 상황에 따라 정책이 **스스로** 정한다.

    a_final = (1 − g(o))·a_BT + g(o)·a_RL,    g(o) = sigmoid(gate_net(o)) ∈ [0, 1]

선형 혼합(hybrid.py)은 α 가 교전 내내 고정이고, 감독형 혼합(shield.py)은 사람이 쓴
안전 조건이 스위치를 켠다. 게이팅형은 "언제 규칙을 더 따르고 언제 학습 정책을 더
따를지"를 작은 게이트 신경망(20-8-1, 177개 파라미터)이 관측을 보고 결정한다.

시간을 아끼는 설계: a_RL 은 **이미 학습된 순수 학습 정책 체크포인트**를 그대로 쓰고,
새로 학습하는 것은 게이트 하나뿐이다. 체크포인트는 학습 정책 가중치를 함께 담아
자체 완결적으로 저장한다.

혼합 수식(뱅크 채널의 원형 평균 포함)은 hybrid.py 와 같다.
"""
from __future__ import annotations
import math
import numpy as np

from .base import Policy
from .bt import BTPolicy
from .hybrid import MLPPolicy
from ..config import OBS_DIM


class GateNet:
    """20 → hidden → 1, tanh 은닉, 시그모이드 출력. numpy 순전파."""

    def __init__(self, hidden=(8,), params: np.ndarray | None = None, seed: int = 0):
        self.hidden = tuple(hidden)
        self.sizes = (OBS_DIM,) + self.hidden + (1,)
        self.n_params = sum(self.sizes[i] * self.sizes[i + 1] + self.sizes[i + 1]
                            for i in range(len(self.sizes) - 1))
        if params is None:
            rng = np.random.default_rng(seed)
            params = rng.normal(0.0, 0.05, self.n_params)      # g ≈ 0.5 에서 출발
        self.set_params(np.asarray(params, dtype=np.float64))

    def set_params(self, flat: np.ndarray) -> None:
        assert flat.size == self.n_params, f"expected {self.n_params}, got {flat.size}"
        self.flat = flat.copy()
        self.W, self.b = [], []
        i = 0
        for a, b in zip(self.sizes[:-1], self.sizes[1:]):
            self.W.append(flat[i:i + a * b].reshape(a, b)); i += a * b
            self.b.append(flat[i:i + b]); i += b

    def __call__(self, obs: np.ndarray) -> float:
        x = np.asarray(obs, dtype=np.float64)
        for k in range(len(self.W) - 1):
            x = np.tanh(x @ self.W[k] + self.b[k])
        y = float((x @ self.W[-1] + self.b[-1])[0])
        return 1.0 / (1.0 + math.exp(-y))


class GatingPolicy(Policy):
    def __init__(self, rl: MLPPolicy, bt_version: int = 2, gate_hidden=(8,),
                 gate_params: np.ndarray | None = None, seed: int = 0):
        self.rl = rl
        self.bt_version = int(bt_version)
        self.bt = BTPolicy(version=self.bt_version)
        self.gate = GateNet(hidden=gate_hidden, params=gate_params, seed=seed)
        self.name = "GATE"
        self.last_node = ""
        self.last_gate = 0.5

    @property
    def flat(self) -> np.ndarray:
        return self.gate.flat

    @property
    def n_params(self) -> int:
        return self.gate.n_params

    def set_params(self, flat: np.ndarray) -> None:
        self.gate.set_params(flat)

    def act(self, obs: np.ndarray) -> np.ndarray:
        g = self.gate(obs)
        self.last_gate = g
        b = np.asarray(self.bt.act(obs), dtype=np.float64)
        self.last_node = self.bt.last_node
        r = np.asarray(self.rl.act(obs), dtype=np.float64)
        out = np.clip((1.0 - g) * b + g * r, -1.0, 1.0)
        mb, mr = b[0] * math.pi, r[0] * math.pi
        out[0] = math.atan2((1.0 - g) * math.sin(mb) + g * math.sin(mr),
                            (1.0 - g) * math.cos(mb) + g * math.cos(mr)) / math.pi
        return out

    def complexity(self) -> dict:
        d = {"gate_params": int(self.gate.n_params)}
        d.update({f"bt_{k}": v for k, v in self.bt.complexity().items()})
        d.update({f"rl_{k}": v for k, v in self.rl.complexity().items()})
        return d

    def save(self, path: str) -> None:
        np.savez(path, kind=np.array("gating"), gate_flat=self.gate.flat,
                 gate_hidden=np.array(self.gate.hidden),
                 rl_flat=self.rl.flat, rl_hidden=np.array(self.rl.hidden),
                 rl_out_act=np.array(self.rl.out_act), bt_version=np.array(self.bt_version))

    @staticmethod
    def load(path: str) -> "GatingPolicy":
        d = np.load(path)
        assert str(d["kind"]) == "gating", f"{path} 는 게이팅형 체크포인트가 아닙니다"
        rl = MLPPolicy(hidden=tuple(int(x) for x in d["rl_hidden"]), params=d["rl_flat"],
                       out_act=str(d["rl_out_act"]))
        return GatingPolicy(rl, bt_version=int(d["bt_version"]),
                            gate_hidden=tuple(int(x) for x in d["gate_hidden"]),
                            gate_params=d["gate_flat"])
