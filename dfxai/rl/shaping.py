"""
보상 셰이핑.

120초 교전의 승패만으로는 절대 학습이 붙지 않습니다(희소보상 문제).
그래서 교전기하로 만든 포텐셜 함수 Phi(s)를 씁니다.

핵심은 **포텐셜 기반 셰이핑(potential-based shaping)** 형태를 지키는 것입니다.

    r_shaped = r_env + (gamma * Phi(s') - Phi(s))

이 형태는 최적정책을 바꾸지 않는다는 것이 이론적으로 보장됩니다
(Ng, Harada & Russell, 1999). 즉 "셰이핑 때문에 이상한 행동을 학습했다"는
반론을 원천적으로 막을 수 있습니다. 논문 방법 절에 이 근거를 꼭 쓰세요.
"""
from __future__ import annotations
import numpy as np
from ..geometry import denorm

# 포텐셜 가중치 — 논문 부록에 그대로 싣고, 민감도 분석 대상으로 삼으세요.
W_ATA = 1.0      # 내 기수가 상대를 향할수록 좋음 (공격 포지션)
W_AA = 0.6       # 내가 상대의 후방일수록 좋음 (꼬리 물기)
W_RANGE = 0.5    # 기총 유효사거리 근처가 좋음
W_ENERGY = 0.25  # 에너지 우위
R_OPT = 600.0    # 선호 사거리 [m]
R_WIDTH = 500.0


def potential(obs: np.ndarray) -> float:
    """Phi(s). 값이 클수록 유리한 교전기하."""
    d = denorm(obs)
    p_ata = 1.0 - d["ata_total"] / np.pi
    p_aa = 1.0 - d["aa"] / np.pi
    p_r = float(np.exp(-((d["r"] - R_OPT) / R_WIDTH) ** 2))
    p_e = float(np.tanh(d["dEs"] / 1500.0))
    return W_ATA * p_ata + W_AA * p_aa + W_RANGE * p_r + W_ENERGY * p_e


def shaped_step_reward(obs: np.ndarray, next_obs: np.ndarray,
                       dmg_dealt: float, dmg_taken: float,
                       gamma: float = 0.997) -> float:
    """한 스텝 보상 = 환경보상 + 포텐셜 차분."""
    r_env = 0.02 * dmg_dealt - 0.02 * dmg_taken
    return r_env + (gamma * potential(next_obs) - potential(obs))


def terminal_reward(winner: int, outcome: str) -> float:
    """종료 보상. 격추승에 가중치를 더 줍니다."""
    if outcome == "crash" and winner == -1:
        return -15.0            # 자기 지면충돌은 크게 벌점
    if outcome == "collision":
        return -5.0
    if winner == 1:
        return 10.0 if outcome == "gun_kill" else 5.0
    if winner == -1:
        return -10.0 if outcome == "gun_kill" else -5.0
    return 0.0


def episode_fitness(res, mean_potential: float) -> float:
    """ES 학습기가 쓰는 적합도. 그래디언트가 없으므로 에피소드 단위 스칼라 하나면 됩니다."""
    f = 0.0
    f += 0.10 * (res.damage_dealt - res.damage_taken)
    f += terminal_reward(res.winner, res.outcome)
    f += 2.0 * mean_potential
    # 규칙 위반은 학습 단계에서도 약하게 벌점 (규칙 준수 축과 일관성 유지)
    v = res.violations_blue
    f -= 3.0 * (v.get("deck", 0.0) + v.get("over_g", 0.0))
    return float(f)
