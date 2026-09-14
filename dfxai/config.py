"""
전역 설정.

여기 있는 값들은 전부 논문 '방법' 절에 그대로 옮겨 적을 수 있도록
출처/근거를 주석으로 남겨두었습니다. 값을 바꾸면 반드시 논문에도 반영하세요.
"""
from dataclasses import dataclass, field, asdict
import numpy as np

G0 = 9.80665          # 중력가속도 [m/s^2]
RHO0 = 1.225          # 해면 공기밀도 [kg/m^3]


@dataclass
class AircraftConfig:
    """F-16급 단발 전투기의 공개 제원에 기반한 점질량 모델 파라미터."""
    mass: float = 9300.0        # 전투중량 [kg] (연료 50% 가정)
    wing_area: float = 27.87    # 주익 면적 [m^2]
    cd0: float = 0.0200         # 유해항력계수
    k_induced: float = 0.117    # 유도항력 계수 (= 1/(pi*AR*e))
    cl_max: float = 1.60        # 최대 양력계수 (버피트 한계)
    thrust_max_sl: float = 106_000.0   # 해면 최대추력(A/B) [N]
    n_max_struct: float = 9.0   # 구조 한계 하중배수 [g]
    n_min_struct: float = 0.0   # 최소 하중배수 (음의 G는 배제)
    v_min: float = 70.0         # 실속 하한 [m/s]
    v_max: float = 600.0        # 속도 상한 [m/s]

    # 1차 지연 시상수 — 조종면 응답을 무한히 빠르지 않게 만드는 항
    tau_bank: float = 0.35      # 뱅크각 응답 시상수 [s]
    tau_load: float = 0.30      # 하중배수 응답 시상수 [s]
    tau_throttle: float = 1.20  # 엔진 응답 시상수 [s]

    bank_cmd_limit: float = np.pi   # 지령 뱅크각 한계 [rad]. arctan2 전 구간을
    #                             포화 없이 표현하기 위해 +-180deg 로 둡니다.
    #                             포화되면 행동이 사실상 이진값이 되어
    #                             대리모델이 깊이 1로도 재현해 버립니다.


@dataclass
class EngagementConfig:
    """교전 규칙. 항공대 대회 규정(200초, 기총, 시간에 따른 명중범위 확대)을 참고해
    축소 재현한 것이며, 수치는 본 연구 전용으로 재조정했습니다."""
    dt_sim: float = 0.05        # 적분 스텝 [s]
    n_substeps: int = 4         # 결정 1회당 적분 횟수 -> 제어주기 0.2s (5 Hz)
    episode_time: float = 120.0  # 교전 제한시간 [s]

    hp_init: float = 100.0
    gun_damage_rate: float = 25.0   # WEZ 체류 시 초당 피해량 [HP/s]
    gun_range_max: float = 1200.0   # 기총 유효 최대사거리 [m]
    gun_range_min: float = 120.0    # 최소사거리 (이보다 가까우면 미명중)
    cone_deg_start: float = 3.0     # 초반 명중 원추 반각 [deg]
    cone_deg_end: float = 10.0       # 종료 시점 명중 원추 반각 [deg]

    # 안전/교전 규칙 (규칙 준수 축에서 사용)
    hard_deck: float = 1000.0       # 최저 고도 [m] — 침범 시 위반 카운트
    over_g_limit: float = 8.0       # 이 값을 넘는 하중배수는 위반
    min_separation: float = 150.0   # 이보다 가까우면 근접 위반
    collision_range: float = 60.0   # 이보다 가까우면 공중충돌(양측 패배)

    # 초기조건 랜덤화 범위
    init_range: tuple = (1500.0, 3500.0)
    init_alt: tuple = (3000.0, 5000.0)
    init_speed: tuple = (180.0, 260.0)


@dataclass
class ObsConfig:
    """관측 정규화 스케일. BT·RL·하이브리드가 모두 동일한 벡터를 받습니다.

    세 모델이 같은 특징공간을 공유한다는 점이 대리모델 충실도 비교의
    공정성을 담보하는 핵심 장치입니다.
    """
    r_scale: float = 5000.0
    v_scale: float = 300.0
    h_scale: float = 6000.0
    dalt_scale: float = 2000.0
    closure_scale: float = 300.0
    es_scale: float = 3000.0


FEATURE_NAMES = [
    "r_norm",          # 0  상대거리
    "ata_h",           # 1  수평면 내 부호 있는 기수-시선 각 (좌/우)
    "ata_v",           # 2  수직면 내 부호 있는 시선 상하각
    "ata_total",       # 3  보어사이트 이탈각 (WEZ 판정용)
    "aa",              # 4  애스펙트각 (0 = 상대 6시 후방)
    "closure",         # 5  접근률
    "dalt",            # 6  고도차 (자기 - 상대)
    "v_own",           # 7  자기 속도
    "v_tgt",           # 8  상대 속도
    "gamma_own",       # 9  자기 경로각
    "mu_own",          # 10 자기 뱅크각
    "n_own",           # 11 자기 하중배수
    "h_own",           # 12 자기 고도
    "dEs",             # 13 비에너지 차 (자기 - 상대)
    "sin_dpsi",        # 14 상대 기수방위 차 sin
    "cos_dpsi",        # 15 상대 기수방위 차 cos
    "hp_own",          # 16 자기 잔여 HP
    "hp_tgt",          # 17 상대 잔여 HP
    "t_frac",          # 18 경과 시간 비율
    "alpha",           # 19 BT/RL 혼합 계수
]
OBS_DIM = len(FEATURE_NAMES)

ACTION_NAMES = ["bank_cmd", "load_cmd", "throttle_cmd"]
ACTION_DIM = len(ACTION_NAMES)


def config_dump() -> dict:
    """논문 부록에 실을 설정 스냅샷."""
    return {
        "aircraft": asdict(AircraftConfig()),
        "engagement": asdict(EngagementConfig()),
        "obs": asdict(ObsConfig()),
        "feature_names": FEATURE_NAMES,
        "action_names": ACTION_NAMES,
    }
