"""
Tacview ACMI 2.2 (텍스트) 내보내기.

ACMI 는 Tacview 의 공개 기록 포맷입니다. 평평한 지구 좌표 (x북, y동, h)를
기준점 주변의 위경도로 환산하고, 자세는 뱅크(mu)·경로각(gamma)·
기수방위(psi)를 그대로 roll·pitch·yaw 에 실어 보냅니다. 3자유도 모델이라
받음각·옆미끄럼은 없으므로 pitch 는 경로각과 같습니다(이 점은 논문에
그림을 실을 때 각주로 밝히는 편이 정직합니다).

무료판 Tacview 로도 열립니다. 재생·시점전환·거리측정·타임라인 북마크가
전부 동작하므로 심사 시연과 교전 디버깅에 그대로 쓸 수 있습니다.

포맷 요약
---------
    FileType=text/acmi/tacview
    FileVersion=2.2
    0,ReferenceLatitude=... ReferenceLongitude=...   (이후 좌표는 전부 상대값)
    #<시각(초)>
    <객체id>,T=경도|위도|고도|롤|피치|요,Name=...,Color=Blue
    0,Event=Message|<객체id>|<내용>

T 의 각 성분은 직전 프레임과 같으면 비워 둘 수 있습니다(공식 델타 압축).
여기서도 그렇게 줄여서 파일 크기를 1/3 수준으로 만듭니다.
"""
from __future__ import annotations
import math
import os
from datetime import datetime, timezone
from typing import Iterable, Sequence

import numpy as np

# 공군사관학교(청주) 부근. 지형과 무관한 평평한 지구 모델이므로 기준점은
# 그림의 배경일 뿐이고 결과에는 영향을 주지 않습니다.
REF_LAT = 36.72
REF_LON = 127.50

BLUE_ID = "101"
RED_ID = "102"

_M_PER_DEG_LAT = 111_132.95


def _m_per_deg_lon(lat_deg: float) -> float:
    return 111_319.49 * math.cos(math.radians(lat_deg))


def _esc(s: str) -> str:
    """ACMI 속성값 이스케이프. 쉼표·역슬래시·개행이 구분자와 충돌합니다."""
    return (str(s).replace("\\", "\\\\").replace(",", "\\,")
            .replace("\n", "\\n"))


def _fmt(v: float, nd: int) -> str:
    s = f"{v:.{nd}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


def _norm_event(e: Sequence) -> tuple[float, str, str, str]:
    """3원소/4원소 이벤트를 (시각, 주체, 종류, 내용)으로 통일."""
    if len(e) == 4:
        t, who, kind, txt = e
    else:
        t, who, txt = e
        kind = "Message"
    kind = str(kind)
    if kind not in ("Message", "Bookmark", "Debug", "Destroyed",
                    "LeftArea", "Timeout"):
        kind = "Message"
    return float(t), str(who), kind, str(txt)


class _ObjectWriter:
    """객체 1대의 델타 압축 상태."""

    def __init__(self, oid: str, ref_lat: float, ref_lon: float):
        self.oid = oid
        self.ref_lat = ref_lat
        self.mdlon = _m_per_deg_lon(ref_lat)
        self.prev_T: list[str] = [""] * 6
        self.prev_props: dict[str, str] = {}
        self.spawned = False

    def line(self, x: float, y: float, h: float,
             mu: float, gamma: float, psi: float,
             props: dict[str, str]) -> str | None:
        """이번 프레임의 기록 라인. 바뀐 게 없으면 None."""
        T = [
            _fmt(y / self.mdlon, 7),                      # 경도 (동쪽 = y)
            _fmt(x / _M_PER_DEG_LAT, 7),                  # 위도 (북쪽 = x)
            _fmt(h, 1),                                   # 고도 [m]
            _fmt(math.degrees(mu), 1),                    # 롤 = 뱅크각
            _fmt(math.degrees(gamma), 1),                 # 피치 = 경로각
            _fmt(math.degrees(psi) % 360.0, 1),           # 요 = 기수방위
        ]
        if self.spawned:
            parts = [a if a != b else "" for a, b in zip(T, self.prev_T)]
        else:
            parts = T
        self.prev_T = T

        fields = []
        if any(parts):
            fields.append("T=" + "|".join(parts))
        for k, v in props.items():
            if self.prev_props.get(k) != v:
                fields.append(f"{k}={v}")
                self.prev_props[k] = v
        self.spawned = True
        if not fields:
            return None
        return self.oid + "," + ",".join(fields)


def write_acmi(path: str,
               traj: np.ndarray,
               events: Iterable[Sequence] = (),
               blue_name: str = "BLUE",
               red_name: str = "RED",
               title: str = "dfxai dogfight",
               briefing: str = "",
               ref_lat: float = REF_LAT,
               ref_lon: float = REF_LON,
               aircraft_type: str = "F-16C") -> str:
    """궤적 배열을 .acmi 파일로 저장하고 경로를 돌려줍니다.

    traj   : env.TRAJ_COLUMNS 순서의 (N, 24) 배열
    events : (시각, 주체, 내용) 또는 (시각, 주체, 종류, 내용) 시퀀스.
             주체는 "blue"/"red"/"both", 종류는 "Message"/"Bookmark".
    """
    traj = np.asarray(traj, dtype=np.float64)
    if traj.ndim != 2 or traj.shape[1] != 24:
        raise ValueError(f"traj 는 (N,24) 여야 합니다. 받은 형상: {traj.shape}")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    ev = sorted([_norm_event(e) for e in events], key=lambda e: e[0])
    ev_i = 0
    ev_id = {"blue": BLUE_ID, "red": RED_ID, "both": BLUE_ID}

    blue = _ObjectWriter(BLUE_ID, ref_lat, ref_lon)
    red = _ObjectWriter(RED_ID, ref_lat, ref_lon)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    out = [
        "FileType=text/acmi/tacview",
        "FileVersion=2.2",
        "0,DataSource=dfxai (3-DoF point-mass dogfight sim)",
        "0,DataRecorder=dfxai.viz.acmi",
        f"0,ReferenceTime={now}",
        f"0,RecordingTime={now}",
        f"0,Title={_esc(title)}",
        "0,Category=Air-to-Air",
        f"0,Briefing={_esc(briefing or f'{blue_name} vs {red_name}')}",
        f"0,ReferenceLongitude={ref_lon}",
        f"0,ReferenceLatitude={ref_lat}",
    ]

    last_stamp: str | None = None

    def stamp(tv: float) -> None:
        """같은 시각의 `#` 줄이 연달아 나오지 않도록 한 번만 찍습니다."""
        nonlocal last_stamp
        s = _fmt(tv, 2)
        if s != last_stamp:
            out.append("#" + s)
            last_stamp = s

    n = traj.shape[0]
    for i in range(n):
        r = traj[i]
        t = r[0]
        while ev_i < len(ev) and ev[ev_i][0] <= t:
            _t, who, kind, txt = ev[ev_i]
            stamp(_t)
            out.append(f"0,Event={kind}|{ev_id.get(who, BLUE_ID)}|{_esc(txt)}")
            ev_i += 1
        stamp(t)

        fire_b, fire_r = r[22] > 0.5, r[23] > 0.5
        pb = {"TAS": _fmt(r[4], 1), "Health": _fmt(r[10] / 100.0, 3),
              "FocusedTarget": RED_ID if fire_b else "0"}
        pr = {"TAS": _fmt(r[14], 1), "Health": _fmt(r[20] / 100.0, 3),
              "FocusedTarget": BLUE_ID if fire_r else "0"}
        if i == 0:
            pb.update(Type="Air+FixedWing", Name=aircraft_type,
                      Pilot=_esc(blue_name), Color="Blue", Coalition="Blue")
            pr.update(Type="Air+FixedWing", Name=aircraft_type,
                      Pilot=_esc(red_name), Color="Red", Coalition="Red")

        lb = blue.line(r[1], r[2], r[3], r[7], r[6], r[5], pb)
        lr = red.line(r[11], r[12], r[13], r[17], r[16], r[15], pr)
        if lb:
            out.append(lb)
        if lr:
            out.append(lr)

    # 남은 이벤트(종료 직후)와 격추 표시
    last = traj[-1]
    while ev_i < len(ev):
        _t, who, kind, txt = ev[ev_i]
        stamp(max(_t, last[0]))
        out.append(f"0,Event={kind}|{ev_id.get(who, BLUE_ID)}|{_esc(txt)}")
        ev_i += 1
    if last[20] <= 0.0 or last[13] <= 0.0:
        stamp(last[0])
        out.append(f"0,Event=Destroyed|{RED_ID}|")
    if last[10] <= 0.0 or last[3] <= 0.0:
        stamp(last[0])
        out.append(f"0,Event=Destroyed|{BLUE_ID}|")

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out) + "\n")
    return path
