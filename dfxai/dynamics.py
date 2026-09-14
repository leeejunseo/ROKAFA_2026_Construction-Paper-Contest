"""
3자유도(point-mass) 비행역학.

평평한 지구, 무풍, 대칭 정상선회를 가정합니다. 상태변수는
    x(북), y(동), h(고도), V(속도), psi(기수방위), gamma(경로각)
이고, 조종 지령은
    mu_c(뱅크각), n_c(하중배수), thr_c(스로틀)
입니다. 실제 뱅크/하중/스로틀은 1차 지연을 거쳐 지령을 추종합니다.

이 모델을 쓰는 이유는 두 가지입니다.
  1) BFM(기본전투기동) 연구에서 널리 쓰이는 표준 수준의 모델이라
     "장난감 시뮬레이터"라는 지적을 피할 수 있습니다.
  2) 조종 입력이 곧 상위 지령이므로, RL이 조종간을 직접 흔드는 대신
     "얼마나 기울이고 몇 G로 당길지"만 결정하면 됩니다.
     탐색 난이도가 크게 낮아집니다.

구현 메모: step() 은 교전당 수천 번 불리는 핫루프라 numpy 대신 math 스칼라
연산으로 작성했습니다. 수식은 동일하며 결과는 반올림 수준에서 같습니다.
"""
from __future__ import annotations
import math
import numpy as np
from .config import AircraftConfig, G0, RHO0

_PI = math.pi
_TWO_PI = 2.0 * math.pi
_GAMMA_LIM = math.pi / 2.2


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def air_density(h: float) -> float:
    """ISA 대류권 근사 (h <= 11 km)."""
    h = max(0.0, min(h, 11_000.0))
    return RHO0 * (1.0 - 2.25577e-5 * h) ** 4.25588


def max_thrust(h: float, cfg: AircraftConfig) -> float:
    """고도에 따른 최대추력 감소. 밀도비의 0.7승 근사."""
    return cfg.thrust_max_sl * (air_density(h) / RHO0) ** 0.7


def max_load_factor(v: float, h: float, cfg: AircraftConfig) -> float:
    """양력 한계와 구조 한계 중 작은 값.

    이 함수 때문에 '코너속도'가 모델에서 자연스럽게 창발합니다.
    저속에서는 양력 한계가, 고속에서는 구조 한계가 지배합니다.
    """
    q = 0.5 * air_density(h) * v * v          # 동압
    n_lift = q * cfg.wing_area * cfg.cl_max / (cfg.mass * G0)
    return _clip(min(n_lift, cfg.n_max_struct), 0.0, cfg.n_max_struct)


class AircraftState:
    """항공기 1대의 상태."""

    __slots__ = ("x", "y", "h", "v", "psi", "gamma", "mu", "n", "thr", "hp", "alive")

    def __init__(self, x, y, h, v, psi, gamma=0.0, hp=100.0):
        self.x = float(x)
        self.y = float(y)
        self.h = float(h)
        self.v = float(v)
        self.psi = float(psi)
        self.gamma = float(gamma)
        self.mu = 0.0      # 현재 뱅크각 [rad]
        self.n = 1.0       # 현재 하중배수 [g]
        self.thr = 0.8     # 현재 스로틀 [0,1]
        self.hp = float(hp)
        self.alive = True

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y, self.h])

    @property
    def vel(self) -> np.ndarray:
        """속도 벡터 (NED 아님, h가 위쪽 양수)."""
        cg = math.cos(self.gamma)
        return self.v * np.array(
            [cg * math.cos(self.psi), cg * math.sin(self.psi), math.sin(self.gamma)]
        )

    @property
    def specific_energy(self) -> float:
        """비에너지 Es = h + V^2/(2g) [m]. 에너지 기동 판단의 핵심 지표."""
        return self.h + self.v * self.v / (2.0 * G0)

    def copy(self) -> "AircraftState":
        s = AircraftState(self.x, self.y, self.h, self.v, self.psi, self.gamma, self.hp)
        s.mu, s.n, s.thr, s.alive = self.mu, self.n, self.thr, self.alive
        return s


def step(state: AircraftState, cmd, dt: float, cfg: AircraftConfig) -> None:
    """지령 cmd = [mu_c(rad), n_c(g), thr_c(0..1)] 로 dt초 전진 적분 (제자리 수정).

    적분은 명시적 오일러입니다. dt=0.05s에서 충분히 안정적이며,
    더 정밀한 결과가 필요하면 RK4로 교체하십시오(인터페이스 동일).
    """
    mu_c, n_c, thr_c = float(cmd[0]), float(cmd[1]), float(cmd[2])

    # --- 지령 포화 ---
    mu_c = _clip(mu_c, -cfg.bank_cmd_limit, cfg.bank_cmd_limit)
    n_avail = max_load_factor(state.v, state.h, cfg)
    n_c = _clip(n_c, cfg.n_min_struct, n_avail)
    thr_c = _clip(thr_c, 0.0, 1.0)

    # --- 1차 지연 응답 ---
    state.mu += (mu_c - state.mu) * min(1.0, dt / cfg.tau_bank)
    state.n += (n_c - state.n) * min(1.0, dt / cfg.tau_load)
    state.thr += (thr_c - state.thr) * min(1.0, dt / cfg.tau_throttle)

    # --- 공력 ---
    rho = air_density(state.h)
    q = 0.5 * rho * state.v * state.v
    lift = state.n * cfg.mass * G0
    cl = lift / max(q * cfg.wing_area, 1e-6)
    cd = cfg.cd0 + cfg.k_induced * cl * cl
    drag = q * cfg.wing_area * cd
    thrust = state.thr * max_thrust(state.h, cfg)

    # --- 운동방정식 ---
    v = max(state.v, 1.0)
    sg, cg = math.sin(state.gamma), math.cos(state.gamma)
    dv = (thrust - drag) / cfg.mass - G0 * sg
    dgamma = (G0 / v) * (state.n * math.cos(state.mu) - cg)
    dpsi = (G0 / v) * state.n * math.sin(state.mu) / max(cg, 0.2)

    state.x += v * cg * math.cos(state.psi) * dt
    state.y += v * cg * math.sin(state.psi) * dt
    state.h += v * sg * dt
    state.v = _clip(v + dv * dt, cfg.v_min, cfg.v_max)
    state.gamma = _clip(state.gamma + dgamma * dt, -_GAMMA_LIM, _GAMMA_LIM)
    state.psi = (state.psi + dpsi * dt + _PI) % _TWO_PI - _PI
