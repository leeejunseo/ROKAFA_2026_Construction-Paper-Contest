"""
학습 기반 정책(MLP)과 BT/RL 혼합 정책.

혼합 규칙은 단 한 줄입니다.

    a = (1 - alpha) * a_BT + alpha * a_RL

alpha = 0  -> 순수 BT
alpha = 1  -> 순수 RL
그 사이 -> 하이브리드

이 구조가 이 프로젝트의 핵심 설계입니다. 세 모델을 따로 만드는 대신
**다이얼 하나**로 연결해 두면:

  (a) 학습을 한 번만 하면 됩니다. 학습 중 alpha 를 매 에피소드 무작위
      추출하고 관측에 포함시키면, 하나의 정책이 모든 alpha 에서 동작합니다.
  (b) 학습 초반에는 alpha 를 낮게 둬서 BT 가 탐색을 이끌게 할 수 있습니다.
      보조바퀴를 달고 배우는 셈이라 밑바닥 학습보다 훨씬 빨리 수렴합니다.
  (c) 평가 시 alpha 를 스윕하면 파레토 평면의 점이 3개가 아니라
      6~8개가 되어 비로소 '곡선'이라 부를 수 있게 됩니다.
"""
from __future__ import annotations
import math
import numpy as np
from .base import Policy
from ..config import OBS_DIM, ACTION_DIM


# ------------------------------------------------------------------ MLP 정책
class MLPPolicy(Policy):
    """의도적으로 불투명한 블랙박스 정책.

    torch 없이 numpy만으로 순전파합니다. ES 학습기(train_es.py)가 직접 쓰고,
    SB3로 PPO를 돌린 경우에도 가중치를 추출해 이 클래스에 실으면
    동일한 평가 파이프라인을 그대로 탈 수 있습니다.
    """

    def __init__(self, hidden=(32, 32), seed: int = 0, params: np.ndarray | None = None,
                 out_act: str = "tanh"):
        # out_act: 출력층 활성. ES 정책은 tanh, SB3 PPO 에서 변환한 정책은 clip
        # (PPO 는 선형 출력 평균을 환경에서 [-1,1] 로 자르므로 그대로 옮겨야 같은 정책).
        self.out_act = out_act
        self.hidden = tuple(hidden)
        self.sizes = (OBS_DIM,) + self.hidden + (ACTION_DIM,)
        self.n_params = sum(self.sizes[i] * self.sizes[i + 1] + self.sizes[i + 1]
                            for i in range(len(self.sizes) - 1))
        if params is None:
            rng = np.random.default_rng(seed)
            params = rng.normal(0.0, 0.5, self.n_params)
        self.set_params(np.asarray(params, dtype=np.float64))
        self.name = "RL"

    def set_params(self, flat: np.ndarray) -> None:
        assert flat.size == self.n_params, f"expected {self.n_params}, got {flat.size}"
        self.flat = flat.copy()
        self.W, self.b = [], []
        i = 0
        for a, b in zip(self.sizes[:-1], self.sizes[1:]):
            self.W.append(flat[i:i + a * b].reshape(a, b)); i += a * b
            self.b.append(flat[i:i + b]); i += b

    def act(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float64)
        for k in range(len(self.W) - 1):
            x = np.tanh(x @ self.W[k] + self.b[k])
        y = x @ self.W[-1] + self.b[-1]
        return np.clip(y, -1.0, 1.0) if self.out_act == "clip" else np.tanh(y)

    def complexity(self) -> dict:
        return {"nn_params": int(self.n_params), "hidden": str(self.hidden)}

    def save(self, path: str) -> None:
        np.savez(path, flat=self.flat, hidden=np.array(self.hidden),
                 out_act=np.array(self.out_act))

    @staticmethod
    def load(path: str) -> "MLPPolicy":
        d = np.load(path)
        out_act = str(d["out_act"]) if "out_act" in d.files else "tanh"
        return MLPPolicy(hidden=tuple(int(x) for x in d["hidden"]), params=d["flat"],
                         out_act=out_act)


# --------------------------------------------------------------- 혼합 정책
class HybridPolicy(Policy):
    """a = (1-alpha)*a_BT + alpha*a_RL

    alpha 는 관측 벡터의 마지막 원소로도 정책에 전달됩니다
    (config.FEATURE_NAMES 의 'alpha'). 학습 시 무작위화해 두면
    하나의 신경망이 모든 alpha 구간에서 동작합니다.
    """

    def __init__(self, bt: Policy, rl: Policy, alpha: float = 0.5):
        self.bt, self.rl = bt, rl
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.name = f"HYB-a{self.alpha:.2f}"

    def act(self, obs: np.ndarray) -> np.ndarray:
        a = self.alpha
        if a <= 0.0:
            return self.bt.act(obs)
        if a >= 1.0:
            return self.rl.act(obs)
        b, r = self.bt.act(obs), self.rl.act(obs)
        out = np.clip((1.0 - a) * b + a * r, -1.0, 1.0)
        # 뱅크 채널은 각도(±1 = ±180°)라 원형 평균을 써야 합니다. 선형 평균은
        # -170° 와 +170° 를 섞어 0° 를 내는데, 이는 양력벡터를 정반대로 돌립니다.
        mb, mr = b[0] * math.pi, r[0] * math.pi
        out[0] = math.atan2((1.0 - a) * math.sin(mb) + a * math.sin(mr),
                            (1.0 - a) * math.cos(mb) + a * math.cos(mr)) / math.pi
        return out

    def complexity(self) -> dict:
        d = {"alpha": self.alpha}
        d.update({f"bt_{k}": v for k, v in self.bt.complexity().items()})
        d.update({f"rl_{k}": v for k, v in self.rl.complexity().items()})
        return d


def make_policy(kind: str, alpha: float = 1.0, bt_version: int = 3,
                rl_path: str | None = None, rl_policy: Policy | None = None) -> Policy:
    """평가 스크립트가 쓰는 팩토리.

    kind: "bt" | "rl" | "hybrid"
    """
    from .bt import BTPolicy
    if kind == "bt":
        return BTPolicy(version=bt_version)
    rl = rl_policy if rl_policy is not None else MLPPolicy.load(rl_path)
    if kind == "rl":
        return rl
    return HybridPolicy(BTPolicy(version=bt_version), rl, alpha=alpha)
