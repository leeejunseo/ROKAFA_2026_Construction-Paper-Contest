"""모든 정책이 따르는 최소 인터페이스."""
from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod


class Policy(ABC):
    name: str = "policy"

    @abstractmethod
    def act(self, obs: np.ndarray) -> np.ndarray:
        """관측 -> 행동 [-1,1]^3 (bank_cmd, load_cmd, throttle_cmd)."""

    def complexity(self) -> dict:
        """모델 복잡도. BT는 노드 수, 신경망은 파라미터 수.

        '어느 쪽에 더 공들였나'를 보고하기 위한 대리 지표입니다.
        논문 결과표에 반드시 함께 실으세요.
        """
        return {}

    def reset(self) -> None:
        pass
