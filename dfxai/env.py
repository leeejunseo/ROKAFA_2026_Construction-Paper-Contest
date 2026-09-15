"""
1대1 기총 도그파이트 교전 환경.

승패 판정:
  - 상대 HP가 0 이하 -> 승
  - 지면 충돌(고도 <= 0) -> 패
  - 공중충돌(상대거리 < collision_range) -> 양측 패(무승부 처리)
  - 제한시간 종료 -> 잔여 HP가 많은 쪽 승, 동일하면 무승부

규칙 준수 축(논문의 세 번째 평가축)은 스텝당 위반율로 집계합니다.
  - 하드덱(최저고도) 침범
  - 과도한 G
  - 최소이격 침범
"""
from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional

from .config import AircraftConfig, EngagementConfig, ObsConfig, ACTION_DIM
from .dynamics import AircraftState, step as dyn_step, max_load_factor
from .geometry import build_obs, relative_geometry, los_range, boresight_angle


def action_to_command(a: np.ndarray, state: AircraftState,
                      ac: AircraftConfig) -> np.ndarray:
    """정책 출력 a in [-1,1]^3 를 물리 지령으로 변환.

    a[0] -> 뱅크각 지령   [-100deg, +100deg]
    a[1] -> 하중배수 지령 [0, 현재 가용 최대 G]
    a[2] -> 스로틀 지령   [0, 1]

    세 모델이 모두 이 동일한 3차원 지령공간에서 행동하므로,
    "BT는 원래 트리라서 설명하기 쉬운 것 아니냐"는 반론이
    행동공간 차이 때문이라는 가능성은 배제됩니다.
    """
    a0 = min(max(float(a[0]), -1.0), 1.0)
    a1 = min(max(float(a[1]), -1.0), 1.0)
    a2 = min(max(float(a[2]), -1.0), 1.0)
    n_avail = max_load_factor(state.v, state.h, ac)
    return (a0 * ac.bank_cmd_limit, 0.5 * (a1 + 1.0) * n_avail, 0.5 * (a2 + 1.0))


def cone_halfangle(t_frac: float, ec: EngagementConfig) -> float:
    """명중 원추 반각. 대회 규정처럼 시간이 갈수록 넓어집니다."""
    d = ec.cone_deg_start + (ec.cone_deg_end - ec.cone_deg_start) * t_frac
    return math.radians(d)


# 궤적 기록 1프레임의 열 이름. viz/acmi.py 가 이 순서를 그대로 씁니다.
TRAJ_COLUMNS = (
    "t",
    "bx", "by", "bh", "bv", "bpsi", "bgamma", "bmu", "bn", "bthr", "bhp",
    "rx", "ry", "rh", "rv", "rpsi", "rgamma", "rmu", "rn", "rthr", "rhp",
    "range", "fire_blue", "fire_red",
)


@dataclass
class EpisodeResult:
    winner: int                 # 1 = 자기(파랑), -1 = 상대(빨강), 0 = 무승부
    outcome: str                # "gun_kill" / "timeout" / "crash" / "collision"
    duration: float
    hp_blue: float
    hp_red: float
    damage_dealt: float
    damage_taken: float
    wez_time_blue: float
    wez_time_red: float
    violations_blue: dict = field(default_factory=dict)
    violations_red: dict = field(default_factory=dict)
    steps: int = 0
    mean_es_blue: float = 0.0


class DogfightEnv:
    """단일 교전을 실행하는 환경.

    Gymnasium 인터페이스가 필요하면 rl/gym_wrapper.py 를 쓰세요.
    여기서는 의존성을 최소화하기 위해 순수 numpy로 유지합니다.
    """

    def __init__(self, ac: AircraftConfig | None = None,
                 ec: EngagementConfig | None = None,
                 oc: ObsConfig | None = None,
                 record_traj: bool = False, traj_stride: int = 2):
        """record_traj : 매 적분 스텝의 월드 상태를 기록합니다(재생/시각화용).

        traj_stride 는 몇 적분 스텝마다 1프레임을 남길지입니다. 기본 2면
        0.1초 간격이라 Tacview 재생이 매끄럽고 파일도 작습니다. 기록이
        꺼져 있으면 본실험 루프에는 분기 하나만 추가될 뿐입니다.
        """
        self.ac = ac or AircraftConfig()
        self.ec = ec or EngagementConfig()
        self.oc = oc or ObsConfig()
        self.dt = self.ec.dt_sim * self.ec.n_substeps
        self.max_steps = int(self.ec.episode_time / self.dt)
        self._rec = bool(record_traj)
        self.traj_stride = max(1, int(traj_stride))
        self.traj: list[tuple] = []
        self.events: list[tuple] = []

    # ------------------------------------------------------------------ 초기화
    def reset(self, seed: int = 0, alpha: float = 1.0,
              alpha_red: float = 0.0):
        """대칭 초기조건.

        두 기체의 고도와 속도를 동일하게 두고, 각자의 초기 기수방위만
        독립적으로 무작위화합니다. 잔여 편향은 평가 단계의 **진영 교대 대전**
        (같은 시드로 청/홍을 바꿔 한 번 더 싸움)으로 완전히 상쇄됩니다.
        이 짝지은 설계 덕분에 초기조건 우연이 승률 비교를 오염시키지 않습니다.
        """
        rng = np.random.default_rng(seed)
        ec = self.ec
        r0 = rng.uniform(*ec.init_range)
        h0 = rng.uniform(*ec.init_alt)
        v0 = rng.uniform(*ec.init_speed)
        bearing = rng.uniform(-np.pi, np.pi)
        psi_b = rng.uniform(-np.pi, np.pi)
        psi_r = rng.uniform(-np.pi, np.pi)
        self.blue = AircraftState(0.0, 0.0, h0, v0, psi_b, hp=ec.hp_init)
        self.red = AircraftState(r0 * np.cos(bearing), r0 * np.sin(bearing),
                                 h0, v0, psi_r, hp=ec.hp_init)
        self.t = 0.0
        self.n_step = 0
        self.alpha = float(alpha)
        self.alpha_red = float(alpha_red)
        self._viol_b = dict(deck=0, over_g=0, separation=0)
        self._viol_r = dict(deck=0, over_g=0, separation=0)
        self._wez_b = 0.0
        self._wez_r = 0.0
        self._dmg_dealt = 0.0
        self._dmg_taken = 0.0
        self._es_sum = 0.0
        self._done = False
        self._outcome = "timeout"
        self.traj = []
        self.events = []
        self._sub = 0
        self._ev = dict(fire_b=False, fire_r=False, deck_b=False, deck_r=False,
                        og_b=False, og_r=False, sep=False)
        if self._rec:
            self._push_frame(r0, False, False)
        return self.obs_blue(), self.obs_red()

    # ------------------------------------------------------------ 궤적 기록
    def _push_frame(self, r: float, fire_b: bool, fire_r: bool) -> None:
        b, d = self.blue, self.red
        self.traj.append((
            self.t,
            b.x, b.y, b.h, b.v, b.psi, b.gamma, b.mu, b.n, b.thr, max(0.0, b.hp),
            d.x, d.y, d.h, d.v, d.psi, d.gamma, d.mu, d.n, d.thr, max(0.0, d.hp),
            r, float(fire_b), float(fire_r),
        ))

    def _edge(self, key: str, active: bool, actor: str, text: str,
              kind: str = "Message") -> None:
        """상승 에지에서만 이벤트를 남깁니다(스텝마다 찍으면 타임라인이 묻힙니다).

        kind="Bookmark" 인 이벤트는 Tacview 타임라인에 표식으로 박혀
        해당 시점으로 바로 점프할 수 있습니다. 눈으로 확인할 가치가 있는
        순간(WEZ 진입·하드덱 침범·종료)에만 씁니다.
        """
        if active and not self._ev[key]:
            self.events.append((self.t, actor, kind, text))
        self._ev[key] = active

    # ------------------------------------------------------------------ 관측
    def obs_blue(self) -> np.ndarray:
        return build_obs(self.blue, self.red, self.t / self.ec.episode_time,
                         self.alpha, self.oc)

    def obs_red(self) -> np.ndarray:
        return build_obs(self.red, self.blue, self.t / self.ec.episode_time,
                         self.alpha_red, self.oc)

    # ------------------------------------------------------------------ 진행
    def step(self, a_blue: np.ndarray, a_red: np.ndarray):
        """제어주기 1회(= n_substeps 적분) 진행."""
        ec, ac = self.ec, self.ac
        cmd_b = action_to_command(a_blue, self.blue, ac)
        cmd_r = action_to_command(a_red, self.red, ac)

        for _ in range(ec.n_substeps):
            dyn_step(self.blue, cmd_b, ec.dt_sim, ac)
            dyn_step(self.red, cmd_r, ec.dt_sim, ac)
            self.t += ec.dt_sim

            lx, ly, lz, r = los_range(self.blue, self.red)
            t_frac = min(1.0, self.t / ec.episode_time)
            cone = cone_halfangle(t_frac, ec)

            # --- WEZ 판정 (기총) ---
            # 전체 기하 대신 거리와 총각만 계산합니다 (핫루프).
            in_range = ec.gun_range_min <= r <= ec.gun_range_max
            fire_b = bool(in_range and
                          boresight_angle(self.blue, lx, ly, lz, r) <= cone)
            if fire_b:
                dmg = ec.gun_damage_rate * ec.dt_sim
                self.red.hp -= dmg
                self._dmg_dealt += dmg
                self._wez_b += ec.dt_sim
            fire_r = bool(in_range and
                          boresight_angle(self.red, -lx, -ly, -lz, r) <= cone)
            if fire_r:
                dmg = ec.gun_damage_rate * ec.dt_sim
                self.blue.hp -= dmg
                self._dmg_taken += dmg
                self._wez_r += ec.dt_sim

            # --- 규칙 위반 집계 ---
            if self.blue.h < ec.hard_deck:
                self._viol_b["deck"] += 1
            if self.blue.n > ec.over_g_limit:
                self._viol_b["over_g"] += 1
            if self.red.h < ec.hard_deck:
                self._viol_r["deck"] += 1
            if self.red.n > ec.over_g_limit:
                self._viol_r["over_g"] += 1
            if r < ec.min_separation:
                self._viol_b["separation"] += 1
                self._viol_r["separation"] += 1

            # --- 궤적/이벤트 기록 ---
            if self._rec:
                self._edge("fire_b", fire_b, "blue", "청 WEZ 진입 (사격)", "Bookmark")
                self._edge("fire_r", fire_r, "red", "홍 WEZ 진입 (사격)", "Bookmark")
                self._edge("deck_b", self.blue.h < ec.hard_deck, "blue",
                           "청 하드덱 침범", "Bookmark")
                self._edge("deck_r", self.red.h < ec.hard_deck, "red",
                           "홍 하드덱 침범", "Bookmark")
                self._edge("og_b", self.blue.n > ec.over_g_limit, "blue", "청 과G")
                self._edge("og_r", self.red.n > ec.over_g_limit, "red", "홍 과G")
                self._edge("sep", r < ec.min_separation, "both", "최소이격 침범")
                self._sub += 1
                if self._sub % self.traj_stride == 0:
                    self._push_frame(r, fire_b, fire_r)

            # --- 종료 조건 ---
            if r < ec.collision_range:
                self._finish(0, "collision"); return self._term()
            if self.blue.h <= 0.0:
                self._finish(-1, "crash"); return self._term()
            if self.red.h <= 0.0:
                self._finish(1, "crash"); return self._term()
            if self.red.hp <= 0.0:
                self._finish(1, "gun_kill"); return self._term()
            if self.blue.hp <= 0.0:
                self._finish(-1, "gun_kill"); return self._term()

        self._es_sum += self.blue.specific_energy
        self.n_step += 1
        if self.t >= ec.episode_time:
            if ec.timeout_rule == "draw":
                self._finish(0, "timeout")
            elif self.blue.hp > self.red.hp:
                self._finish(1, "timeout")
            elif self.red.hp > self.blue.hp:
                self._finish(-1, "timeout")
            else:
                self._finish(0, "timeout")
        return self._term()

    def _finish(self, winner: int, outcome: str):
        self._done = True
        self._winner = winner
        self._outcome = outcome
        if self._rec:
            # 종료 프레임은 stride 와 무관하게 반드시 남깁니다
            # (직전 프레임과 시각이 같으면 중복이므로 건너뜁니다).
            if not self.traj or self.traj[-1][0] < self.t:
                self._push_frame(relative_geometry(self.blue, self.red)["r"],
                                 False, False)
            who = {1: "blue", -1: "red", 0: "both"}[winner]
            label = {1: "청 승", -1: "홍 승", 0: "무승부"}[winner]
            self.events.append((self.t, who, "Bookmark", f"종료: {outcome} ({label})"))

    def _term(self):
        return self.obs_blue(), self.obs_red(), self._done

    # ------------------------------------------------------------------ 결과
    def result(self) -> EpisodeResult:
        n_sub = max(1, self.n_step * self.ec.n_substeps)
        return EpisodeResult(
            winner=getattr(self, "_winner", 0),
            outcome=self._outcome,
            duration=self.t,
            hp_blue=max(0.0, self.blue.hp),
            hp_red=max(0.0, self.red.hp),
            damage_dealt=self._dmg_dealt,
            damage_taken=self._dmg_taken,
            wez_time_blue=self._wez_b,
            wez_time_red=self._wez_r,
            violations_blue={k: v / n_sub for k, v in self._viol_b.items()},
            violations_red={k: v / n_sub for k, v in self._viol_r.items()},
            steps=self.n_step,
            mean_es_blue=self._es_sum / max(1, self.n_step),
        )


def run_episode(blue_policy, red_policy, seed: int = 0, alpha: float = 1.0,
                alpha_red: float = 0.0, env: Optional[DogfightEnv] = None,
                record: bool = False):
    """교전 1회 실행.

    record=True 이면 (관측, 행동) 궤적을 함께 반환합니다.
    이 궤적이 대리모델 충실도 계산의 입력이 됩니다.
    """
    env = env or DogfightEnv()
    ob, orr = env.reset(seed=seed, alpha=alpha, alpha_red=alpha_red)
    obs_log, act_log = [], []
    for _ in range(env.max_steps + 1):
        a_b = blue_policy.act(ob)
        a_r = red_policy.act(orr)
        if record:
            obs_log.append(ob.copy())
            act_log.append(np.asarray(a_b, dtype=np.float64).copy())
        ob, orr, done = env.step(a_b, a_r)
        if done:
            break
    res = env.result()
    if record:
        return res, np.asarray(obs_log), np.asarray(act_log)
    return res
