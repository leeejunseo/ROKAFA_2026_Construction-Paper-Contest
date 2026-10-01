"""
JSBSim F-16 기체 + 내부 추종 제어기 (6자유도 평가 전용 교차검증).

설계 원칙 (설계안: yoonjun/논문_수정제안/6DOF_교차검증_설계안.md)
  정책·관측·교전 규칙은 3자유도(dfxai) 것을 그대로 두고 **비행역학만** 바꾼다.
  정책이 내는 지령은 3자유도와 같은 [뱅크각 mu_c, 하중배수 n_c, 스로틀 thr_c] 이고,
  이 모듈의 추종 제어기가 이를 조종간 입력으로 바꿔 JSBSim F-16 을 조종한다.

구성
  JSBSim F-16 (jsbsim 패키지 내장 aircraft/f16)
    - 기체 자체의 비행제어계(FBW)를 그대로 쓴다: 옆 조종간 = 롤 속도 지령,
      앞뒤 조종간 = 하중 지령, 받음각 제한 포함. FBW 는 기체의 일부로 본다.
  TrackingController (새로 구현, "조종사" 역할)
    - 뱅크: 속도축 뱅크 오차 -> 롤 속도 요구 -> 옆 조종간 (FBW 의 롤 속도 지령 척도 그대로)
      오차는 최단 경로로 감는다(3자유도는 1차 지연이라 ±180° 경계에서 먼 길로 돎).
    - 하중배수: 오차 PI -> 앞뒤 조종간 (적분 포화 방지)
    - 방향타: 0 (FBW 요 댐퍼에 맡김),  스로틀: 그대로 전달 (0.99 초과 = 후기연소)

상태 변환 (JSBSim -> AircraftState, 3자유도와 같은 정의)
  x, y : 적도·본초자오선 기준 국지 평면 [m] (교전 범위 수 km 에서 오차 무시 가능)
  v    : 진대기속도,  psi : 속도 방위각(지상 궤적),  gamma : 비행경로각
  mu   : **속도축 뱅크각** = 속도 벡터 둘레로 양력(기체 -z축) 방향이 기운 각
  n    : 기체 수직 하중배수 (accelerations/Nz)
"""
from __future__ import annotations
import math

from ..config import AircraftConfig
from ..dynamics import AircraftState

FT = 0.3048
_A = 6_378_137.0                      # WGS84 장반경
_E2 = 6.69437999014e-3
_M0 = _A * (1 - _E2)                  # 적도에서 자오선 곡률반경
_N0 = _A                              # 적도에서 묘유선 곡률반경


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class TrackingController:
    """[mu_c, n_c, thr_c] -> (옆 조종간, 앞뒤 조종간, 방향타, 스로틀).

    이득은 dfxai.sixdof.step_response 의 계단 응답으로 맞췄다. 목표는 3자유도의
    1차 지연(뱅크 0.35 s, 하중 0.30 s)에 가까운 응답이지만, 기체와 FBW 가 허용하는
    한계(롤 속도, 받음각, 하중)를 넘지는 않는다. 그 차이가 곧 교차검증이 재려는 것이다.
    """

    # 이득: 220·260 m/s 계단 응답이 3자유도 응답에 가장 가깝고(평균 제곱 오차) 오버슈트가
    # 작은 조합을 격자 탐색으로 골랐다(뱅크 오버슈트 약 1%, 하중 약 4%).
    def __init__(self, k_mu: float = 5.0, p_max: float = math.radians(240.0),
                 k_p: float = 2.0, kp_n: float = 0.12, ki_n: float = 0.2,
                 kff_n: float = 0.08, k_lim: float = 0.6, n_max: float = 9.0):
        self.k_mu, self.p_max, self.k_p = k_mu, p_max, k_p
        self.kp_n, self.ki_n, self.kff_n = kp_n, ki_n, kff_n
        self.k_lim, self.n_max = k_lim, n_max
        self.i_n = 0.0

    def reset(self):
        self.i_n = 0.0

    def __call__(self, mu_c, n_c, thr_c, mu, p, nz, dt):
        # --- 뱅크 -> 롤 속도 요구 -> 옆 조종간 (FBW: 조종간 1 = 롤 속도 1/0.31821 rad/s) ---
        p_dem = max(-self.p_max, min(self.p_max, self.k_mu * _wrap(mu_c - mu)))
        ail = (p_dem + self.k_p * (p_dem - p)) * 0.31821
        ail = max(-1.0, min(1.0, ail))
        # --- 하중배수 PI (당김 = 음수, FBW 범위 [-1, 0.44]) ---
        n_c = min(n_c, self.n_max)
        e = n_c - nz
        u = -(self.kff_n * (n_c - 1.0) + self.kp_n * e + self.ki_n * self.i_n)
        # G 제한기: 지령을 넘는 하중은 강하게 민다(실제 F-16 비행제어계의 9 G 제한에 해당).
        # 고 G 를 유지한 채 롤을 뒤집을 때 롤-피치 결합으로 하중이 튀는 것을 막는다.
        u += self.k_lim * max(0.0, nz - n_c)
        elev = max(-1.0, min(0.44, u))
        if elev == u or (elev <= -1.0 and e < 0) or (elev >= 0.44 and e > 0):
            self.i_n += e * dt                  # 포화 방향으로는 적분하지 않음
        return ail, elev, 0.0, max(0.0, min(1.0, thr_c))


class JSBAircraft:
    """JSBSim F-16 한 대.

    reset_from() 은 교전마다 JSBSim 인스턴스를 **새로 만든다**. run_ic() 로 초기조건만 다시
    걸면 비행제어계 필터 등의 내부 상태가 이전 교전에서 넘어와, 같은 시드라도 앞에 어떤 교전을
    돌렸는지에 따라 결과가 조금씩 달라졌다(재현성 문제). 모델 재적재는 약 6 ms 로 싸다.
    """

    def __init__(self, model: str = "f16"):
        self.model = model
        self._new_fdm()
        self.dt = self.fdm.get_delta_t()        # 1/120 s
        self.ctrl = TrackingController()
        self.trim_ok = True
        # 연료: 교전마다 처음 양으로 다시 채우고 소모를 멈춘다. 3자유도처럼 질량이 일정하고
        # (약 9,360 kg, 3자유도 9,300 kg), 후기연소로 연료가 바닥나 다음 교전 트림이 실패하는 일도 없다.
        n_tank = 0
        while True:
            try:
                self.fdm[f"propulsion/tank[{n_tank}]/contents-lbs"]
                n_tank += 1
            except KeyError:
                break
        self._fuel0 = [self.fdm[f"propulsion/tank[{i}]/contents-lbs"] for i in range(n_tank)]

    def _new_fdm(self) -> None:
        import jsbsim
        self.fdm = jsbsim.FGFDMExec(None)
        self.fdm.set_debug_level(0)
        self.fdm.load_model(self.model)

    # ------------------------------------------------------------------ 초기화
    def reset_from(self, s: AircraftState) -> None:
        self._new_fdm()
        f = self.fdm
        f["ic/lat-geod-rad"] = s.x / _M0
        f["ic/long-gc-rad"] = s.y / _N0
        f["ic/h-sl-ft"] = s.h / FT
        f["ic/vt-fps"] = s.v / FT
        f["ic/psi-true-rad"] = s.psi
        f["ic/gamma-rad"] = s.gamma
        f["ic/phi-rad"] = 0.0
        for i, q in enumerate(self._fuel0):
            f[f"propulsion/tank[{i}]/contents-lbs"] = q
        f["propulsion/fuel_freeze"] = 1
        f["propulsion/set-running"] = -1
        f.run_ic()
        f["gear/gear-cmd-norm"] = 0.0
        f["fcs/aileron-cmd-norm"] = 0.0
        f["fcs/elevator-cmd-norm"] = 0.0
        f["fcs/rudder-cmd-norm"] = 0.0
        try:
            f["simulation/do_simple_trim"] = 1   # 수평 정상비행 트림
            self.trim_ok = True
        except Exception:                        # pragma: no cover - 드문 초기조건
            self.trim_ok = False
            f["fcs/throttle-cmd-norm"] = 0.8
        self.ctrl.reset()

    # ------------------------------------------------------------------ 상태
    def bank_wind(self) -> float:
        """속도축 뱅크각 [rad]: 3자유도의 mu 와 같은 정의 (120 Hz 핫루프라 math 스칼라로 계산).

        양력 방향 l = -(z_b 에서 속도 성분을 뺀 것), 기준 '위' u = (위쪽 단위벡터의 속도 수직 성분),
        오른쪽 r = v x u 로 두고  mu = atan2(l·r, l·u).  (NED 좌표)
        """
        f = self.fdm
        ph, th, ps = f["attitude/phi-rad"], f["attitude/theta-rad"], f["attitude/psi-rad"]
        cph, sph, cth, sth, cps, sps = (math.cos(ph), math.sin(ph), math.cos(th),
                                        math.sin(th), math.cos(ps), math.sin(ps))
        zx = cph * sth * cps + sph * sps
        zy = cph * sth * sps - sph * cps
        zz = cph * cth
        vx, vy, vz = f["velocities/v-north-fps"], f["velocities/v-east-fps"], f["velocities/v-down-fps"]
        vn = math.sqrt(vx * vx + vy * vy + vz * vz) or 1e-9
        vx, vy, vz = vx / vn, vy / vn, vz / vn
        d = zx * vx + zy * vy + zz * vz
        lx, ly, lz = -(zx - d * vx), -(zy - d * vy), -(zz - d * vz)
        ux, uy, uz = vz * vx, vz * vy, -1.0 + vz * vz        # up=(0,0,-1) 의 속도 수직 성분
        nu = math.sqrt(ux * ux + uy * uy + uz * uz)
        if nu < 1e-6:                            # 수직 비행: 뱅크 정의 불가 -> 기체 롤각
            return _wrap(ph)
        ux, uy, uz = ux / nu, uy / nu, uz / nu
        rx, ry, rz = vy * uz - vz * uy, vz * ux - vx * uz, vx * uy - vy * ux
        return math.atan2(lx * rx + ly * ry + lz * rz, lx * ux + ly * uy + lz * uz)

    def sync(self, s: AircraftState) -> None:
        """JSBSim 상태를 AircraftState 에 옮긴다 (hp, alive 는 교전 환경이 관리)."""
        f = self.fdm
        s.x = f["position/lat-geod-rad"] * _M0
        s.y = f["position/long-gc-rad"] * _N0
        s.h = f["position/h-sl-ft"] * FT
        s.v = f["velocities/vt-fps"] * FT
        s.psi = _wrap(f["flight-path/psi-gt-rad"])
        s.gamma = f["flight-path/gamma-rad"]
        s.mu = self.bank_wind()
        s.n = f["accelerations/Nz"]
        s.thr = f["fcs/throttle-pos-norm"]

    # ------------------------------------------------------------------ 진행
    def step(self, cmd, duration: float) -> None:
        """지령 [mu_c, n_c, thr_c] 를 duration 초 동안 추종 (JSBSim 120 Hz, 제어기도 120 Hz)."""
        f = self.fdm
        mu_c, n_c, thr_c = float(cmd[0]), float(cmd[1]), float(cmd[2])
        n_steps = max(1, int(round(duration / self.dt)))
        for _ in range(n_steps):
            ail, elev, rud, thr = self.ctrl(mu_c, n_c, thr_c, self.bank_wind(),
                                            f["velocities/p-aero-rad_sec"],
                                            f["accelerations/Nz"], self.dt)
            f["fcs/aileron-cmd-norm"] = ail
            f["fcs/elevator-cmd-norm"] = elev
            f["fcs/rudder-cmd-norm"] = rud
            f["fcs/throttle-cmd-norm"] = thr
            f.run()


def reference_state(h: float = 4000.0, v: float = 220.0) -> AircraftState:
    return AircraftState(0.0, 0.0, h, v, 0.0)


__all__ = ["JSBAircraft", "TrackingController", "reference_state", "AircraftConfig"]
