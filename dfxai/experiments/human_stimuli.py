"""
사람 대상 이해도 실험 자극 생성기.

D* 는 "이 정책의 행동을 몇 단계 규칙으로 재현할 수 있는가"라는 기능 기반 지표다.
박사 수준 심사에서는 "그것이 사람의 이해도와 관계있는가"를 묻는다. 이 스크립트는
그 실험에 쓸 문항을 정책 로그에서 자동으로 뽑는다. 실험 자체(참가자 모집·응답
수집)는 사람이 해야 하며, 절차는 paper/사람평가_프로토콜.md 에 있다.

문항 형식 (다음 기동 예측)
--------------------------
참가자는 정책 하나의 교전 장면 K 개를 보고(학습 단계), 새 장면 N 개에서 그 정책이
다음 0.2초에 낼 지령을 예측한다(시험 단계).
  * 상황: 거리, 좌우 총각, 상하 총각, 애스펙트각, 접근률, 고도차, 자기 고도, 속도,
          잔여 HP, 경과 시간 — 관측 벡터를 물리 단위로 풀어 쓴 것
  * 응답: 뱅크 방향 {좌, 수평, 우} × 하중배수 {낮음 <3G, 중간 3–6G, 높음 >6G}
  * 채점: 정책의 실제 지령을 같은 구간으로 나눈 정답과 일치 여부

    python -m dfxai.experiments.human_stimuli --outdir results/main \
        --conds BT-v2 BT-v3 RL-s0-b300-a1.00 RL-s1-b300-a1.00 --n-train 12 --n-test 20

출력: results/human_study/<cond>_items.csv, items_all.md (인쇄용)
"""
from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd

from ..geometry import denorm
from ..dynamics import max_load_factor
from ..config import AircraftConfig

_AC = AircraftConfig()


def _bank_class(a0: float) -> str:
    deg = a0 * 180.0
    return "수평" if abs(deg) < 15 else ("우" if deg > 0 else "좌")


def _g_class(a1: float, v: float, h: float) -> str:
    n = 0.5 * (a1 + 1.0) * max_load_factor(v, h, _AC)
    return "낮음(<3G)" if n < 3 else ("중간(3–6G)" if n < 6 else "높음(>6G)")


def describe(obs: np.ndarray) -> str:
    d = denorm(obs)
    side = "좌" if d["ata_h"] < 0 else "우"
    updown = "위" if d["ata_v"] > 0 else "아래"
    return (f"상대 거리 {d['r']:.0f} m, 내 기수에서 {side} {abs(np.degrees(d['ata_h'])):.0f}° "
            f"{updown} {abs(np.degrees(d['ata_v'])):.0f}°, 애스펙트각 {np.degrees(d['aa']):.0f}° "
            f"(0°=내가 상대 후방), 접근률 {d['closure']:+.0f} m/s, 상대보다 {d['dalt']:+.0f} m, "
            f"내 고도 {d['h_own']:.0f} m, 속도 {d['v_own']:.0f} m/s, 내 뱅크 {np.degrees(d['mu_own']):+.0f}°, "
            f"HP 나 {d['hp_own']:.0f} / 상대 {d['hp_tgt']:.0f}, 경과 {d['t_frac']*120:.0f} s")


def make_items(outdir: str, cond: str, n_train: int, n_test: int, seed: int) -> pd.DataFrame:
    d = np.load(os.path.join(outdir, f"traj_{cond}.npz"))
    obs, act = d["obs"], d["act"]
    rng = np.random.default_rng(seed)
    # 사거리 3 km 안, 교전 중반 이후 장면을 우선 (초반 접근 장면은 정보량이 적음)
    ok = np.where((obs[:, 0] * 5000 < 3000) & (obs[:, 18] > 0.1))[0]
    idx = rng.choice(ok, n_train + n_test, replace=False)
    rows = []
    for k, i in enumerate(idx):
        o, a = obs[i], act[i]
        dd = denorm(o)
        rows.append(dict(cond=cond, phase="학습" if k < n_train else "시험", item=k + 1,
                         situation=describe(o), answer_bank=_bank_class(a[0]),
                         answer_g=_g_class(a[1], dd["v_own"], dd["h_own"]),
                         raw_bank_deg=a[0] * 180.0, raw_load_cmd=a[1], row_index=int(i)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="사람 이해도 실험 문항 생성")
    ap.add_argument("--outdir", default="results/main")
    ap.add_argument("--conds", nargs="+", required=True)
    ap.add_argument("--n-train", type=int, default=12)
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/human_study")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    md = ["# 사람 이해도 실험 문항 (자동 생성)\n"]
    for c in a.conds:
        df = make_items(a.outdir, c, a.n_train, a.n_test, a.seed)
        df.to_csv(os.path.join(a.out, f"{c}_items.csv"), index=False)
        md.append(f"\n## 정책 {c}\n")
        for _, r in df.iterrows():
            ans = f" → 정답: 뱅크 {r['answer_bank']}, G {r['answer_g']}" if r["phase"] == "학습" else ""
            md.append(f"- [{r['phase']} {r['item']}] {r['situation']}{ans}")
    with open(os.path.join(a.out, "items_all.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"저장: {a.out}")


if __name__ == "__main__":
    main()
