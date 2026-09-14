"""
상대기하 계산과 관측 벡터 생성.

관측을 원시 상태(위치·속도 6+6차원)가 아니라 교전기하 특징으로 압축하는 것이
샘플 효율을 크게 끌어올립니다. 도그파이트는 거리·ATA·AA·에너지차
좌표계에서 사실상 완전히 기술되기 때문입니다.

동시에, BT와 RL이 **똑같은 이 벡터**를 입력으로 받도록 설계했습니다.
그래야 대리모델 충실도를 같은 특징공간에서 비교할 수 있습니다.
"""
from __future__ import annotations
import numpy as np
from .config import ObsConfig, OBS_DIM
from .dynamics import AircraftState


def relative_geometry(own: AircraftState, tgt: AircraftState) -> dict:
    """교전기하 원시값(정규화 전)을 딕셔너리로 반환."""
    los = tgt.pos - own.pos                      # 시선 벡터 (자기 -> 상대)
    r = float(np.linalg.norm(los))
    los_u = los / max(r, 1e-6)

    v_own = own.vel
    v_tgt = tgt.vel
    u_own = v_own / max(np.linalg.norm(v_own), 1e-6)

    # ATA(Antenna Train Angle): 자기 기수(속도벡터)와 시선 사이의 각
    ata_total = float(np.arccos(np.clip(np.dot(u_own, los_u), -1.0, 1.0)))

    # 수평 성분: 좌선회/우선회 판단에 필요하므로 부호를 살립니다.
    brg = np.arctan2(los[1], los[0])
    ata_h = float((brg - own.psi + np.pi) % (2 * np.pi) - np.pi)

    # 수직 성분: 시선의 상하각에서 자기 경로각을 뺀 값
    horiz = float(np.hypot(los[0], los[1]))
    ata_v = float(np.arctan2(los[2], max(horiz, 1e-6)) - own.gamma)
    ata_v = float(np.clip(ata_v, -np.pi / 2, np.pi / 2))

    # AA(Aspect Angle): 상대 꼬리에서 나를 바라본 각. 0 = 내가 상대의 정확한 6시.
    # 상대의 꼬리방향(-u_tgt)과 '상대->나' 방향(-los_u) 사이의 각과 동일합니다.
    # AA=0 이면 내가 상대의 정확한 6시, AA=pi 이면 내가 상대의 정면.
    u_tgt = v_tgt / max(np.linalg.norm(v_tgt), 1e-6)
    aa = float(np.arccos(np.clip(np.dot(u_tgt, los_u), -1.0, 1.0)))

    closure = float(-np.dot(v_tgt - v_own, los_u))  # 양수면 접근 중
    dpsi = float((tgt.psi - own.psi + np.pi) % (2 * np.pi) - np.pi)

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
    o = np.empty(OBS_DIM, dtype=np.float64)
    o[0] = np.clip(g["r"] / cfg.r_scale, 0.0, 2.0)
    o[1] = g["ata_h"] / np.pi
    o[2] = g["ata_v"] / (np.pi / 2)
    o[3] = g["ata_total"] / np.pi
    o[4] = g["aa"] / np.pi
    o[5] = np.clip(g["closure"] / cfg.closure_scale, -3.0, 3.0)
    o[6] = np.clip(g["dalt"] / cfg.dalt_scale, -3.0, 3.0)
    o[7] = own.v / cfg.v_scale
    o[8] = tgt.v / cfg.v_scale
    o[9] = own.gamma / (np.pi / 2)
    o[10] = own.mu / np.pi
    o[11] = own.n / 9.0
    o[12] = own.h / cfg.h_scale
    o[13] = np.clip(g["dEs"] / cfg.es_scale, -3.0, 3.0)
    o[14] = np.sin(g["dpsi"])
    o[15] = np.cos(g["dpsi"])
    o[16] = own.hp / 100.0
    o[17] = tgt.hp / 100.0
    o[18] = t_frac
    o[19] = alpha
    return o


def denorm(obs: np.ndarray, cfg: ObsConfig | None = None) -> dict:
    """관측 벡터를 물리단위로 되돌립니다. BT가 사람이 읽는 조건문을 쓰기 위해 필요."""
    cfg = cfg or ObsConfig()
    return dict(
        r=obs[0] * cfg.r_scale,
        ata_h=obs[1] * np.pi,
        ata_v=obs[2] * (np.pi / 2),
        ata_total=obs[3] * np.pi,
        aa=obs[4] * np.pi,
        closure=obs[5] * cfg.closure_scale,
        dalt=obs[6] * cfg.dalt_scale,
        v_own=obs[7] * cfg.v_scale,
        v_tgt=obs[8] * cfg.v_scale,
        gamma_own=obs[9] * (np.pi / 2),
        mu_own=obs[10] * np.pi,
        n_own=obs[11] * 9.0,
        h_own=obs[12] * cfg.h_scale,
        dEs=obs[13] * cfg.es_scale,
        dpsi=float(np.arctan2(obs[14], obs[15])),
        hp_own=obs[16] * 100.0,
        hp_tgt=obs[17] * 100.0,
        t_frac=obs[18],
        alpha=obs[19],
    )
