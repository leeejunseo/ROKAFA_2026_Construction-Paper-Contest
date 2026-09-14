"""
상대기하 계산과 관측 벡터 생성.

관측을 원시 상태(위치·속도 6+6차원)가 아니라 교전기하 특징으로 압축하는 것이
샘플 효율을 크게 끌어올립니다. 도그파이트는 거리·ATA·AA·에너지차
좌표계에서 사실상 완전히 기술되기 때문입니다.

동시에, BT와 RL이 **똑같은 이 벡터**를 입력으로 받도록 설계했습니다.
그래야 대리모델 충실도를 같은 특징공간에서 비교할 수 있습니다.

구현 메모: 이 모듈은 적분 스텝마다 여러 번 호출되는 핫루프입니다. 3원소
벡터에 numpy 를 쓰면 호출 오버헤드가 계산보다 커지므로, 수식은 그대로 두고
math 모듈 스칼라 연산으로 작성했습니다(결과는 부동소수 반올림 수준에서 동일).
"""
from __future__ import annotations
import math
import numpy as np
from .config import ObsConfig, OBS_DIM
from .dynamics import AircraftState

_PI = math.pi
_TWO_PI = 2.0 * math.pi
_HALF_PI = math.pi / 2.0


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def _unit_velocity(s: AircraftState) -> tuple[float, float, float]:
    """속도벡터의 단위벡터 (x북, y동, h상)."""
    cg = math.cos(s.gamma)
    vx = s.v * (cg * math.cos(s.psi))
    vy = s.v * (cg * math.sin(s.psi))
    vz = s.v * math.sin(s.gamma)
    n = max(math.sqrt(vx * vx + vy * vy + vz * vz), 1e-6)
    return vx / n, vy / n, vz / n


def los_range(own: AircraftState, tgt: AircraftState) -> tuple[float, float, float, float]:
    """시선벡터(자기 -> 상대)와 거리."""
    lx, ly, lz = tgt.x - own.x, tgt.y - own.y, tgt.h - own.h
    return lx, ly, lz, math.sqrt(lx * lx + ly * ly + lz * lz)


def boresight_angle(own: AircraftState, lx: float, ly: float, lz: float,
                    r: float) -> float:
    """ATA(총각): 자기 속도벡터와 시선 사이의 각 [rad]. WEZ 판정용 경량 경로."""
    rr = max(r, 1e-6)
    ux, uy, uz = _unit_velocity(own)
    d = (ux * lx + uy * ly + uz * lz) / rr
    return math.acos(_clip(d, -1.0, 1.0))


def relative_geometry(own: AircraftState, tgt: AircraftState) -> dict:
    """교전기하 원시값(정규화 전)을 딕셔너리로 반환."""
    lx, ly, lz, r = los_range(own, tgt)
    rr = max(r, 1e-6)
    ex, ey, ez = lx / rr, ly / rr, lz / rr          # 시선 단위벡터

    uox, uoy, uoz = _unit_velocity(own)
    # ATA(Antenna Train Angle): 자기 기수(속도벡터)와 시선 사이의 각
    ata_total = math.acos(_clip(uox * ex + uoy * ey + uoz * ez, -1.0, 1.0))

    # 수평 성분: 좌선회/우선회 판단에 필요하므로 부호를 살립니다.
    brg = math.atan2(ly, lx)
    ata_h = (brg - own.psi + _PI) % _TWO_PI - _PI

    # 수직 성분: 시선의 상하각에서 자기 경로각을 뺀 값
    horiz = math.hypot(lx, ly)
    ata_v = _clip(math.atan2(lz, max(horiz, 1e-6)) - own.gamma, -_HALF_PI, _HALF_PI)

    # AA(Aspect Angle): 상대 꼬리에서 나를 바라본 각. 0 = 내가 상대의 정확한 6시.
    # 상대의 꼬리방향(-u_tgt)과 '상대->나' 방향(-los_u) 사이의 각과 동일합니다.
    # AA=0 이면 내가 상대의 정확한 6시, AA=pi 이면 내가 상대의 정면.
    utx, uty, utz = _unit_velocity(tgt)
    aa = math.acos(_clip(utx * ex + uty * ey + utz * ez, -1.0, 1.0))

    # 접근률: 양수면 접근 중 (원식: -dot(v_tgt - v_own, los_u))
    vo = own.v
    vt = tgt.v
    closure = -((utx * vt - uox * vo) * ex + (uty * vt - uoy * vo) * ey
                + (utz * vt - uoz * vo) * ez)
    dpsi = (tgt.psi - own.psi + _PI) % _TWO_PI - _PI

    return dict(
        r=r, ata_total=ata_total, ata_h=ata_h, ata_v=ata_v, aa=aa,
        closure=closure, dalt=own.h - tgt.h, dpsi=dpsi,
        dEs=own.specific_energy - tgt.specific_energy,
    )


def build_obs(own: AircraftState, tgt: AircraftState, t_frac: float,
              alpha: float, cfg: ObsConfig | None = None) -> np.ndarray:
    """정규화된 관측 벡터 (config.FEATURE_NAMES 순서와 반드시 일치)."""
    cfg = cfg or ObsConfig()
    g = relative_geometry(own, tgt)
    return np.array((
        _clip(g["r"] / cfg.r_scale, 0.0, 2.0),
        g["ata_h"] / _PI,
        g["ata_v"] / _HALF_PI,
        g["ata_total"] / _PI,
        g["aa"] / _PI,
        _clip(g["closure"] / cfg.closure_scale, -3.0, 3.0),
        _clip(g["dalt"] / cfg.dalt_scale, -3.0, 3.0),
        own.v / cfg.v_scale,
        tgt.v / cfg.v_scale,
        own.gamma / _HALF_PI,
        own.mu / _PI,
        own.n / 9.0,
        own.h / cfg.h_scale,
        _clip(g["dEs"] / cfg.es_scale, -3.0, 3.0),
        math.sin(g["dpsi"]),
        math.cos(g["dpsi"]),
        own.hp / 100.0,
        tgt.hp / 100.0,
        t_frac,
        alpha,
    ), dtype=np.float64)


def denorm(obs: np.ndarray, cfg: ObsConfig | None = None) -> dict:
    """관측 벡터를 물리단위로 되돌립니다. BT가 사람이 읽는 조건문을 쓰기 위해 필요."""
    cfg = cfg or ObsConfig()
    o = obs.tolist() if hasattr(obs, "tolist") else list(obs)
    return dict(
        r=o[0] * cfg.r_scale,
        ata_h=o[1] * _PI,
        ata_v=o[2] * _HALF_PI,
        ata_total=o[3] * _PI,
        aa=o[4] * _PI,
        closure=o[5] * cfg.closure_scale,
        dalt=o[6] * cfg.dalt_scale,
        v_own=o[7] * cfg.v_scale,
        v_tgt=o[8] * cfg.v_scale,
        gamma_own=o[9] * _HALF_PI,
        mu_own=o[10] * _PI,
        n_own=o[11] * 9.0,
        h_own=o[12] * cfg.h_scale,
        dEs=o[13] * cfg.es_scale,
        dpsi=math.atan2(o[14], o[15]),
        hp_own=o[16] * 100.0,
        hp_tgt=o[17] * 100.0,
        t_frac=o[18],
        alpha=o[19],
    )
