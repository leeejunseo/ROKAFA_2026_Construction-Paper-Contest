"""
교범 기반 기준선 행동트리 BT-M / BT-M+ (C안 H1).

근거 교범
    Korean Air Force TTP 3-3 Vol.5 「Basic Employment Manual F-16C」(2005.10.1, UNCLASSIFIED)
    4.3 기본전투기동(BFM) 개념, 4.4 공격 BFM, 4.5 방어 BFM, 4.6 정면(High-Aspect) BFM.
    같은 계열의 RNoAF AFTTP 3-3 Vol.5(2001)에도 같은 내용이 있는지는 대응표에 따로 표시했다.
    규칙과 교범 절의 대응은 docs/BT-M_교범대응표.md.

두 가지 판
    BT-M  (기준선, 22노드) : 교범 BFM 의 뼈대 — 안전(통과 이격·하드덱 회복), 방어(브레이크 턴),
                            공격(기총 리드·선회원 당김·선회원 진입), 기본 선회(리드 턴/2서클),
                            코너 플래토 속도 관리. 하이브리드의 규칙 쪽 기준점으로 쓴다.
    BT-M+ (상세판, 32노드) : BT-M + 기총 회피 저크, 거리 회복, 오버슛 래그, 리드 시 아이들,
                            양력벡터 편향(상대 POM 위·기수 하향 당김), 진입 속도창, 등가대기속도 판단.
    두 판은 같은 BTMParams 상수를 쓰고, 상세 규칙 켜짐 여부(use_*)만 다르다.

옮기지 않은 것 (시뮬레이터에 없는 요소)
    미사일 사격·회피, 채프/플레어(IRCM), 레이더 락, 시각 단서(캐노피 위 LOS 이동 등),
    "IR 미사일 때문에 아이들 출력으로 브레이크" 같은 미사일 전용 기법.
    1서클(반경) 싸움 선택(4.6.9.2): 상대의 선회 방향을 관측 20개로 알 수 없어 상태 없는 규칙으로
    만들면 시험(300교전)에서 점수가 떨어졌다. 같은 기체끼리는 교범 4.6.40(Equal Adversary)대로
    교착이 기본이라 2서클 선회율 싸움을 기본으로 둔다.

견고성 원칙 (실험 환경이 바뀌어도 고칠 곳이 BTMParams 하나가 되도록)
    1) 문턱값은 물리 단위(m, m/s, deg, G). 교범의 ft·KCAS 값과 변환 근거를 주석에 남긴다.
    2) 하드덱·과G 한계·기총 사거리는 EngagementConfig 에서 읽는다.
    3) 관측은 denorm() 의 이름으로만 읽는다.
    4) 하중 지령은 절대 G 로 계산한 뒤 [-1,1] 로 바꾼다(저속·6자유도에서도 같은 의미).

출력은 다른 정책과 같은 [-1,1]^3 (bank_cmd, load_cmd, throttle_cmd).
"""
from __future__ import annotations
import math
from dataclasses import dataclass, asdict, replace

import numpy as np

from .base import Policy
from .bt import Node, Selector, Sequence, Condition, Action
from ..config import AircraftConfig, EngagementConfig, RHO0
from ..dynamics import air_density, max_load_factor
from ..geometry import denorm

FT = 0.3048          # ft -> m
KT = 0.514444        # kt -> m/s
_D = math.radians


def _clip(x, lo=-1.0, hi=1.0):
    return lo if x < lo else (hi if x > hi else x)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# =============================================================== 상수(조정 지점)
@dataclass
class BTMParams:
    """BT-M 의 모든 조정 상수. 주석 괄호 안은 교범 원값과 절 번호."""

    # --- 안전 (SAFETY 하위트리: 감독형 혼합이 그대로 가져다 쓴다) -------------
    g_margin: float = 0.3                 # 규칙 과G 한계(8 G)보다 이만큼 낮게 자율 제한 [G]
    #                                       (4.5.13 "as hard as body/stores limits allow" -> 규칙 한계에 여유)
    bubble_r: float = 500.0 * FT          # 통과 최소 이격 [m] (4.6.5.1 "500 feet for continuation training";
    #                                       시뮬레이터 최소이격 규칙 150 m 와 같은 크기)
    cpa_time: float = 2.5                 # 최근접점까지 이 시간 안이고 예측 이격이 버블 안이면 회피 [s]
    pass_aa: float = _D(90.0)             # 정면 조우: 내가 상대 3/9 선 앞(AA>90°)이고
    pass_ata: float = _D(90.0)            #            상대가 내 3/9 선 앞(ATA<90°)일 때만
    pass_bank: float = _D(75.0)           # 회피 뱅크
    pass_g: float = 6.0                   # 회피 하중 [G]
    pass_roll_tol: float = _D(30.0)       # 실제 뱅크가 회피 뱅크에서 이 각 안에 들어오면 당김, 그 전에는 1 G 로 굴리기만
    floor_lookahead: float = 5.0          # 하드덱 예측 시간 [s]
    floor_margin: float = 300.0           # 예측 고도가 하드덱 + 이 값 아래면 회복 [m] (약 1,000 ft)
    floor_zone: float = 4000.0 * FT       # 하드덱 위 이 고도 안에서는 기수 하향 선회 금지 [m] (4.6.32 "less than 4,000 feet")

    # --- 에너지 관리 (4.3.3.7 코너 플래토 330~440 KCAS, 4.6.12) ----------------
    v_rate_hi: float = 440.0 * KT         # 최대 선회율 속도창 상한 (등가대기속도) [m/s]
    thr_fast: float = 0.30                # 속도창 위에서의 스로틀 (감속)
    speed_ref_alt: float = 4000.0         # >= 0: 속도 문턱값을 이 고도의 진대기속도로 고정 환산해 v_own 과 비교
    #                                       (초기 고도대 3~5 km 의 중앙). < 0: 매 순간 등가대기속도로 비교.

    # --- 방어 BFM (4.5) --------------------------------------------------------
    def_threat_ata: float = _D(45.0)      # 상대 기수가 나를 이 각 안으로 겨누고 (4.3.3.6 WEZ 최대 ATA 45° 기준)
    def_my_ata: float = _D(90.0)          # 상대가 내 3/9 선 뒤에 있고
    def_range: float = 9000.0 * FT        # 거리가 9,000 ft 안이면 방어 (4.4.4 9,000 ft set)

    # --- 공격 BFM (4.4) --------------------------------------------------------
    off_aa: float = _D(90.0)              # 내가 상대 후방 반구(AA<90°)이고
    off_ata: float = _D(90.0)             # 내 기수가 상대 쪽(ATA<90°)이면 공격
    tc_range: float = 6000.0 * FT         # 선회원 크기 [m] (4.4.4 반경 ~3,000 ft -> 선회원 6,000 ft)
    entry_g: float = 4.0                  # 선회원 밖 접근 하중 [G] (4.4.4.3 least-moving parts, 에너지 보존)
    gun_range: float = 4500.0 * FT        # 리드 전환 거리 [m] (4.4.16 3,500~4,500 ft)
    gun_ata: float = _D(40.0)             # 상대가 기수 40° 안 (4.4.16 "30 to 40 degrees out the top of the HUD")
    lead_gain: float = 1.3                # 리드추격 이득 (1 = 순수추격)

    # --- 상세판(BT-M+) 전용 규칙과 상수 ---------------------------------------
    use_jink: bool = False                # 기총 회피 롤링 턱언더 저크 (4.5.36)
    jink_ata: float = _D(15.0)            #   상대 ATA 가 이 안이고
    jink_range_scale: float = 1.1         #   거리가 기총 최대사거리의 이 배 안이며
    jink_alt: float = 2000.0 * FT         #   하드덱 위 고도 여유가 있을 때 (4.5.36.1.1 1,500~2,000 ft)
    jink_roll: float = _D(90.0)           #   현재 뱅크에서 같은 방향으로 더 감는 각
    jink_relax_bank: float = _D(135.0)    #   이 뱅크를 넘으면 당김을 늦춤 (4.5.36.1.2 "approximately 135 degrees")
    jink_relax_g: float = 2.0
    use_range_control: bool = False       # 선회원 안에서 너무 가까워지면 당김 완화 (4.4.5.3)
    range_floor: float = 4000.0 * FT
    relax_g: float = 4.0
    use_overshoot_lag: bool = False       # 최소거리 안 접근 시 래그 + 아이들 (4.4.16 shoot-lag-shoot)
    overshoot_range: float = 1000.0 * FT
    lag_gain: float = 0.3
    use_closure_idle: bool = False        # 리드 확보 후 접근률 크면 아이들 (4.4.16 "pull power to idle")
    idle_closure: float = 30.0
    idle_range: float = 3000.0 * FT
    use_lv_bias: bool = False             # 양력벡터 편향: 선회원 당김은 상대 POM 약간 위(4.4.5.2),
    tc_lv_bias: float = _D(3.0)           #   기본 선회는 약간 아래 nose-low pull(4.6.18.2),
    rate_lv_bias: float = _D(-3.0)        #   진입 속도 초과 시 2~3° 상승(4.4.4.2)
    entry_climb_bias: float = _D(3.0)
    use_entry_speed: bool = False         # 선회원 진입 속도창 450~480 KCAS (4.4.4.2)
    entry_v_lo: float = 450.0 * KT
    entry_v_hi: float = 480.0 * KT

    @classmethod
    def core(cls, **kw) -> "BTMParams":
        """BT-M 기준선."""
        return cls(**kw)

    @classmethod
    def full(cls, **kw) -> "BTMParams":
        """BT-M+ 상세판: 모든 상세 규칙 켬, 속도는 매 순간 등가대기속도로 판단."""
        base = dict(use_jink=True, use_range_control=True, use_overshoot_lag=True,
                    use_closure_idle=True, use_lv_bias=True, use_entry_speed=True,
                    speed_ref_alt=-1.0)
        base.update(kw)
        return cls(**base)

    def as_dict(self) -> dict:
        ang = ("_ata", "_aa", "_bank", "_bias", "_roll")
        return {k: (round(math.degrees(v), 3) if isinstance(v, float) and k.endswith(ang) else
                    (round(v, 3) if isinstance(v, float) else v))
                for k, v in asdict(self).items()}


# =============================================================== 보조 계산
_AC = AircraftConfig()


def _eas(v: float, h: float) -> float:
    """등가대기속도 [m/s]. 교범의 KCAS 문턱값과 비교하려고 쓴다(저마하에서 CAS≈EAS)."""
    return v * math.sqrt(air_density(h) / RHO0)


def _load_to_action(g_cmd: float, bb: dict) -> float:
    """절대 하중 지령 [G] -> load_cmd [-1,1]. 가용 G 보다 크면 1(최대)로 포화."""
    n_av = max(bb["n_avail"], 1e-3)
    return _clip(2.0 * min(1.0, max(0.0, g_cmd) / n_av) - 1.0)


def _plateau_throttle(bb: dict, P: BTMParams) -> float:
    """코너 플래토 유지: 440 KCAS 를 넘으면 감속, 아니면 최대 (4.3.3.7, 4.6.12)."""
    return P.thr_fast if bb["v_eas"] > P.v_rate_hi else 1.0


def _set(bb: dict, mu: float, g: float, thr: float) -> None:
    bb["a"][0] = _clip(_wrap(mu) / _AC.bank_cmd_limit)
    bb["a"][1] = _load_to_action(g, bb)
    bb["a"][2] = _clip(2.0 * thr - 1.0)


def _pursuit_bank(bb: dict, P: BTMParams, lead: float = 1.0, lv_bias: float = 0.0) -> float:
    """양력벡터를 상대 쪽으로 (lead>1 리드, <1 래그, lv_bias>0 상대 POM 위).

    편향은 선회 중일 때만 건다: 상대가 기수 정면에 있을 때 편향만 남으면 atan2 가 배면(±180°)
    뱅크를 내는 불연속이 생기므로 수평 이탈각 30° 까지 비례해서 키운다.
    바닥 근처(4.6.32)에서는 기수 하향 선회(|mu|>90°)를 수평 선회로 제한한다.
    """
    ah = bb["ata_h"] * lead
    av = bb["ata_v"] + lv_bias * min(1.0, abs(ah) / _D(30.0))
    mu = 0.0 if (abs(ah) < 1e-3 and abs(av) < 1e-3) else math.atan2(ah, av)
    if bb["h_deck"] < P.floor_zone and abs(mu) > _D(90.0):
        mu = math.copysign(_D(90.0), mu)
    return mu


# =============================================================== 트리
def _build(P: BTMParams, g_cap: float) -> tuple[Node, Node]:
    """(전체 트리, 안전 하위트리). 꺼진 상세 규칙은 노드 자체를 만들지 않는다(노드 수가 실제와 같도록)."""
    bias = P.use_lv_bias

    # ---------------- SAFETY ----------------
    def pass_inside_bubble(bb):
        """정면 조우에서 최근접점(CPA) 예측 이격이 버블 안인가. 관측만으로 계산(상대 경로각 0 가정)."""
        if bb["closure"] <= 0.0 or bb["aa"] < P.pass_aa or bb["ata_total"] > P.pass_ata:
            return False
        cg = math.cos(bb["gamma_own"])
        rh = bb["r"] * math.cos(bb["ata_v"] + bb["gamma_own"])
        px, py, pz = rh * math.cos(bb["ata_h"]), rh * math.sin(bb["ata_h"]), -bb["dalt"]
        vx = bb["v_tgt"] * math.cos(bb["dpsi"]) - bb["v_own"] * cg
        vy = bb["v_tgt"] * math.sin(bb["dpsi"])
        vz = -bb["v_own"] * math.sin(bb["gamma_own"])
        vv = vx * vx + vy * vy + vz * vz
        if vv < 1e-6:
            return False
        t = -(px * vx + py * vy + pz * vz) / vv
        if t <= 0.0 or t > P.cpa_time:
            return False
        mx, my, mz = px + vx * t, py + vy * t, pz + vz * t
        bb["cpa_side"] = my
        return math.sqrt(mx * mx + my * my + mz * mz) < P.bubble_r

    def lateral_offset(bb):
        side = -1.0 if bb.get("cpa_side", 0.0) > 0.0 else 1.0   # 상대가 지나갈 쪽의 반대 (같으면 우측)
        mu_cmd = side * P.pass_bank
        # 하중을 푼 채로 먼저 굴려 양력벡터를 회피 방향에 맞춘 뒤 당긴다 (4.4.5.2 "unloaded roll to set Lv").
        # 6자유도에서 배면에 가까운 자세로 바로 당기면 상대 고도 쪽으로 기수가 떨어져 충돌했다.
        g = P.pass_g if abs(_wrap(mu_cmd - bb["mu_own"])) < P.pass_roll_tol else 1.0
        _set(bb, mu_cmd, g, bb["thr_plateau"])

    def floor_predicted(bb):
        h_pred = bb["h_own"] + bb["v_own"] * math.sin(bb["gamma_own"]) * P.floor_lookahead
        return min(bb["h_own"], h_pred) < bb["deck"] + P.floor_margin

    def recover_lift_vector_up(bb):
        _set(bb, 0.0, g_cap, 1.0)

    safety = Selector("SAFETY", [
        Sequence("pass_deconfliction", [Condition("pass_inside_bubble", pass_inside_bubble),
                                        Action("lateral_offset", lateral_offset)]),
        Sequence("floor_recovery", [Condition("floor_predicted", floor_predicted),
                                    Action("recover_lift_vector_up", recover_lift_vector_up)]),
    ])

    # ---------------- DEFENSIVE (4.5) ----------------
    def threatened(bb):
        return (bb["thr_ata"] < P.def_threat_ata and bb["ata_total"] > P.def_my_ata
                and bb["r"] < P.def_range)

    def break_turn(bb):
        _set(bb, _pursuit_bank(bb, P, 1.0), g_cap, bb["thr_plateau"])

    def_children = []
    if P.use_jink:
        def in_enemy_gun_wez(bb):
            return (bb["thr_ata"] < P.jink_ata and bb["r"] < P.jink_range_scale * bb["gun_rmax"]
                    and bb["h_deck"] > P.jink_alt)

        def rolling_tuck_under(bb):
            mu = bb["mu_own"]
            side = 1.0 if mu >= 0.0 else -1.0               # 지금 선회 방향으로 계속 굴린다
            g = P.jink_relax_g if abs(mu) > P.jink_relax_bank else g_cap
            _set(bb, mu + side * P.jink_roll, g, bb["thr_plateau"])

        def_children.append(Sequence("guns_defense", [
            Condition("in_enemy_gun_wez", in_enemy_gun_wez),
            Action("rolling_tuck_under_jink", rolling_tuck_under)]))
    def_children.append(Action("break_turn_into_threat", break_turn))
    defensive = Sequence("DEFENSIVE", [
        Condition("threatened", threatened),
        Selector("defensive_select", def_children) if len(def_children) > 1 else def_children[0],
    ])

    # ---------------- OFFENSIVE (4.4) ----------------
    def offensive_position(bb):
        return bb["aa"] < P.off_aa and bb["ata_total"] < P.off_ata

    def gun_lead_window(bb):
        return bb["r"] < P.gun_range and bb["ata_total"] < P.gun_ata

    def lead_pursuit_gun(bb):
        thr = bb["thr_plateau"]
        if P.use_closure_idle and bb["closure"] > P.idle_closure and bb["r"] < P.idle_range:
            thr = 0.0
        _set(bb, _pursuit_bank(bb, P, P.lead_gain), g_cap, thr)

    def inside_turn_circle(bb):
        return bb["r"] < P.tc_range

    def high_g_pull(bb):
        _set(bb, _pursuit_bank(bb, P, 1.0, P.tc_lv_bias if bias else 0.0), g_cap, bb["thr_plateau"])

    def turn_circle_entry(bb):
        thr, b = bb["thr_plateau"], 0.0
        if P.use_entry_speed:
            v = bb["v_eas"]
            thr = 1.0 if v < P.entry_v_lo else (0.6 if v <= P.entry_v_hi else P.thr_fast)
            if bias and v > P.entry_v_hi:
                b = P.entry_climb_bias
        _set(bb, _pursuit_bank(bb, P, 1.0, b), P.entry_g, thr)

    off_children = []
    if P.use_overshoot_lag:
        def inside_min_range_closing(bb):
            return bb["r"] < P.overshoot_range and bb["closure"] > 0.0

        def lag_reposition(bb):
            _set(bb, _pursuit_bank(bb, P, P.lag_gain), g_cap, 0.0)

        off_children.append(Sequence("overshoot_control", [
            Condition("inside_min_range_closing", inside_min_range_closing),
            Action("lag_idle_reposition", lag_reposition)]))
    off_children.append(Sequence("gunshot", [Condition("gun_lead_window", gun_lead_window),
                                             Action("lead_pursuit_gun", lead_pursuit_gun)]))
    if P.use_range_control:
        def range_too_close(bb):
            return bb["r"] < P.range_floor and bb["closure"] > 0.0

        def relax_pull(bb):
            _set(bb, _pursuit_bank(bb, P, 1.0, P.tc_lv_bias if bias else 0.0), P.relax_g, bb["thr_plateau"])

        off_children.append(Sequence("range_control", [Condition("range_too_close", range_too_close),
                                                       Action("relax_pull", relax_pull)]))
    off_children += [Sequence("at_turn_circle", [Condition("inside_turn_circle", inside_turn_circle),
                                                 Action("high_g_pull", high_g_pull)]),
                     Action("turn_circle_entry", turn_circle_entry)]
    offensive = Sequence("OFFENSIVE", [Condition("offensive_position", offensive_position),
                                       Selector("offensive_select", off_children)])

    # ---------------- 기본: 리드 턴 / 2서클 선회율 싸움 (4.6.5, 4.6.9.1) ----------------
    def rate_turn(bb):
        b = P.rate_lv_bias if (bias and bb["h_deck"] > P.floor_zone) else 0.0
        _set(bb, _pursuit_bank(bb, P, 1.0, b), g_cap, bb["thr_plateau"])

    root = Selector("root_BT-M", [safety, defensive, offensive,
                                  Action("rate_turn_lead_or_two_circle", rate_turn)])
    return root, safety


# =============================================================== 정책
SAFETY_NODES = ("lateral_offset", "recover_lift_vector_up")


class ManualBTPolicy(Policy):
    """교범 기반 행동트리. ManualBTPolicy() = BT-M, ManualBTPolicy.full() = BT-M+."""

    def __init__(self, params: BTMParams | None = None,
                 ec: EngagementConfig | None = None, ac: AircraftConfig | None = None,
                 name: str | None = None):
        self.P = params or BTMParams.core()
        self.ec = ec or EngagementConfig()
        self.ac = ac or _AC
        self.g_cap = self.ec.over_g_limit - self.P.g_margin
        self.root, self.safety = _build(self.P, self.g_cap)
        detailed = any(getattr(self.P, k) for k in ("use_jink", "use_range_control", "use_overshoot_lag",
                                                     "use_closure_idle", "use_lv_bias", "use_entry_speed"))
        self.name = name or ("BT-M+" if detailed else "BT-M")
        self.last_node = ""

    @classmethod
    def full(cls, **kw) -> "ManualBTPolicy":
        return cls(BTMParams.full(**kw))

    def _blackboard(self, obs: np.ndarray) -> dict:
        bb = denorm(obs)
        bb["a"] = [0.0, 0.0, 1.0]
        bb["deck"] = self.ec.hard_deck
        bb["gun_rmax"] = self.ec.gun_range_max
        bb["h_deck"] = bb["h_own"] - self.ec.hard_deck
        bb["thr_ata"] = math.pi - bb["aa"]             # 상대 기수가 나를 겨누는 각 (0 = 정조준)
        h_ref = self.P.speed_ref_alt if self.P.speed_ref_alt >= 0.0 else bb["h_own"]
        bb["v_eas"] = _eas(bb["v_own"], h_ref)
        bb["n_avail"] = max_load_factor(bb["v_own"], bb["h_own"], self.ac)
        bb["thr_plateau"] = _plateau_throttle(bb, self.P)
        bb["fired_node"] = ""
        return bb

    def act(self, obs: np.ndarray) -> np.ndarray:
        bb = self._blackboard(obs)
        self.root.tick(bb)
        self.last_node = bb["fired_node"]
        a = bb["a"]
        return np.array([_clip(a[0]), _clip(a[1]), _clip(a[2])])

    def safety_act(self, obs: np.ndarray):
        """안전 하위트리만 실행. 발동하면 (지령, 노드 이름), 아니면 (None, "") — 감독형 혼합용."""
        bb = self._blackboard(obs)
        if self.safety.tick(bb):
            a = bb["a"]
            return np.array([_clip(a[0]), _clip(a[1]), _clip(a[2])]), bb["fired_node"]
        return None, ""

    def complexity(self) -> dict:
        return {"bt_nodes": self.root.count(), "bt_version": self.name,
                "safety_nodes": self.safety.count(), "g_cap": self.g_cap}

    def describe(self) -> str:
        return self.root.describe()
