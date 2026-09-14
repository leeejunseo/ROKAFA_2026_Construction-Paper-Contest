"""
보상 셰이핑.

120초 교전의 승패만으로는 절대 학습이 붙지 않습니다(희소보상 문제).
그래서 교전기하로 만든 포텐셜 함수 Phi(s)를 씁니다.

핵심은 **포텐셜 기반 셰이핑(potential-based shaping)** 형태를 지키는 것입니다.

    r_shaped = r_env + (gamma * Phi(s') - Phi(s))

이 형태는 최적정책을 바꾸지 않는다는 것이 이론적으로 보장됩니다
(Ng, Harada & Russell, 1999). 즉 "셰이핑 때문에 이상한 행동을 학습했다"는
반론을 원천적으로 막을 수 있습니다. 논문 방법 절에 이 근거를 꼭 쓰세요.

설계 이력 (논문 '방법' 또는 부록에 그대로 쓸 수 있는 실패 기록)
------------------------------------------------------------------
1차 설계는 Phi = ATA항 + AA항 + 사거리항 + 에너지항 이었고, ES 적합도에
2.0 x 평균 Phi 를 더했습니다. 그 결과 학습 정책은 **상대와 6~9 km 떨어져
고도 5~8 km 로 올라가 기수만 상대 쪽으로 둔 채 도망다니는** 정책으로 수렴했습니다
(WEZ 체류 0초, 피해 0, 100% 시간종료). 원거리에서도 ATA 가 작고 에너지 우위가
커서 포텐셜만으로 적합도 1.5~2.0 을 챙길 수 있었기 때문입니다. 전형적인 보상
해킹입니다. 그래서 다음 세 가지를 바꿨습니다.

  (a) 기하 항(ATA·AA·에너지)에 근접 가중 prox(r) 을 곱해, 기총 사거리 밖에서는
      포텐셜이 빠르게 사라지게 했습니다. 상대를 향해 접근해야만 점수가 납니다.
  (b) 적합도에서 포텐셜 비중을 2.0 -> 0.5 로 줄이고, 실제 교전 지표
      (WEZ 체류시간, 피해량)의 비중을 키웠습니다.
  (c) 양측 피해가 전혀 없는 시간종료(회피 무승부)에 벌점을 뒀습니다.
      대회 규정이 소극적 교전을 불리하게 다루는 것과 같은 취지입니다.

포텐셜 형태를 바꿔도 (a) 는 여전히 상태만의 함수이므로 셰이핑 정리의 전제는
유지됩니다. (b)(c) 는 ES 적합도(에피소드 스칼라) 설계이며 논문에 명시합니다.
"""
from __future__ import annotations
import math
import numpy as np
from ..geometry import denorm

# 포텐셜 가중치 — 논문 부록에 그대로 싣고, 민감도 분석 대상으로 삼으세요.
W_ATA = 1.0      # 내 기수가 상대를 향할수록 좋음 (공격 포지션)
W_AA = 0.6       # 내가 상대의 후방일수록 좋음 (꼬리 물기)
W_RANGE = 0.5    # 기총 유효사거리 근처가 좋음
W_ENERGY = 0.15  # 에너지 우위 (근접 시에만)
R_OPT = 600.0    # 선호 사거리 [m]
R_WIDTH = 500.0
R_PROX = 1200.0  # 이 거리(기총 최대사거리)까지는 근접 가중 1, 그 밖에서 지수 감쇠
R_DECAY = 1000.0

# 적합도 가중치
K_DAMAGE = 0.20      # HP 1점당 (최대 +-20)
K_WEZ = 0.50         # WEZ 체류 1초당
K_POTENTIAL = 0.50   # 평균 포텐셜
K_VIOLATION = 3.0    # 스텝당 위반율 (하드덱 + 과G)
PASSIVE_DRAW = -8.0  # 양측 무피해 시간종료 (회피 정책이 격추당하는 것과 비슷한 수준으로 불리하게)


def proximity(r: float) -> float:
    """근접 가중. 사거리 안이면 1, 밖에서는 R_DECAY 마다 e 배 감쇠."""
    return 1.0 if r <= R_PROX else math.exp(-(r - R_PROX) / R_DECAY)


def potential(obs: np.ndarray) -> float:
    """Phi(s). 값이 클수록 유리한 교전기하. 사거리 밖에서는 0 에 가깝습니다."""
    d = denorm(obs)
    prox = proximity(d["r"])
    p_ata = 1.0 - d["ata_total"] / np.pi
    p_aa = 1.0 - d["aa"] / np.pi
    p_r = math.exp(-((d["r"] - R_OPT) / R_WIDTH) ** 2)
    p_e = math.tanh(d["dEs"] / 1500.0)
    return prox * (W_ATA * p_ata + W_AA * p_aa + W_ENERGY * p_e) + W_RANGE * p_r


def shaped_step_reward(obs: np.ndarray, next_obs: np.ndarray,
                       dmg_dealt: float, dmg_taken: float,
                       gamma: float = 0.997) -> float:
    """한 스텝 보상 = 환경보상 + 포텐셜 차분. (PPO 경로용)"""
    r_env = 0.02 * dmg_dealt - 0.02 * dmg_taken
    return r_env + (gamma * potential(next_obs) - potential(obs))


# 적합도 프리셋.
#   default    : 본실험. 격추승 +10, 시간종료승 +5.
#   kill_first : 부록 강건성 실험. 본실험에서 학습 정책 일부가 "한 번 쏘고 이탈해
#                시간종료 HP 우위로 이기는" 전술로 수렴했으므로(규칙상 합법),
#                격추를 크게 우대하고 시간종료승을 거의 보상하지 않는 변형에서도
#                성능-설명가능성 관계의 형태가 유지되는지 확인합니다.
PRESETS = {
    "default":    dict(win_kill=10.0, win_time=5.0, k_wez=K_WEZ),
    "kill_first": dict(win_kill=25.0, win_time=1.0, k_wez=1.0),
}


def terminal_reward(winner: int, outcome: str, preset: str = "default") -> float:
    """종료 보상. 격추승에 가중치를 더 줍니다."""
    p = PRESETS[preset]
    if outcome == "crash" and winner == -1:
        return -15.0            # 자기 지면충돌은 크게 벌점
    if outcome == "collision":
        return -5.0
    if winner == 1:
        return p["win_kill"] if outcome == "gun_kill" else p["win_time"]
    if winner == -1:
        return -10.0 if outcome == "gun_kill" else -5.0
    return 0.0


def episode_fitness(res, mean_potential: float, preset: str = "default") -> float:
    """ES 학습기가 쓰는 적합도. 그래디언트가 없으므로 에피소드 단위 스칼라 하나면 됩니다."""
    p = PRESETS[preset]
    f = 0.0
    f += K_DAMAGE * (res.damage_dealt - res.damage_taken)
    f += p["k_wez"] * res.wez_time_blue
    f += terminal_reward(res.winner, res.outcome, preset)
    f += K_POTENTIAL * mean_potential
    if res.outcome == "timeout" and res.damage_dealt == 0.0 and res.damage_taken == 0.0:
        f += PASSIVE_DRAW
    # 규칙 위반은 학습 단계에서도 약하게 벌점 (규칙 준수 축과 일관성 유지)
    v = res.violations_blue
    f -= K_VIOLATION * (v.get("deck", 0.0) + v.get("over_g", 0.0))
    return float(f)
