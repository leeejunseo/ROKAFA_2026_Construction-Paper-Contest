"""
롤 동역학 진단 (R9 보조).

본실험 환경(롤 속도 상한 없음, 뱅크 1차 지연)에서 정책들이 실제로 얼마나 빠르게
롤했는지, 그리고 뱅크 지령이 ±180° 경계를 건너뛸 때 "먼 길"로 도는 경우가 얼마나
되는지를 잽니다. 동역학 코드는 바꾸지 않고 적분 함수를 감싸 기록만 합니다.

  롤 속도       p = (mu_after - mu_before) / dt          (적분 스텝 0.05 s 단위)
  먼 길 롤     |mu_c - mu| > 180°  (최단 경로가 아니라 수평을 거쳐 반대로 도는 상황)

실행:  python -m dfxai.analysis.roll_diag --n-seeds 20 --out results/r9_roll_diag.csv
"""
from __future__ import annotations
import argparse, math
import numpy as np
import pandas as pd

from .. import env as env_mod
from ..env import DogfightEnv, run_episode
from ..agents.bt import BTPolicy
from ..experiments.run_eval import build_condition

_REC: list = []
_BLUE = {"obj": None}
_orig_step = env_mod.dyn_step


def _rec_step(state, cmd, dt, cfg):
    mu0 = state.mu
    mu_c = min(max(float(cmd[0]), -cfg.bank_cmd_limit), cfg.bank_cmd_limit)
    _orig_step(state, cmd, dt, cfg)
    if state is _BLUE["obj"]:
        _REC.append(((state.mu - mu0) / dt, abs(mu_c - mu0) > math.pi))


def conditions() -> list[dict]:
    c = [dict(name=f"BT-v{v}", kind="bt", bt_version=v) for v in (1, 2, 3)]
    c += [dict(name=f"RL-s{s}-b300-a1.00", kind="rl", alpha=1.0,
               ckpt=f"results/es/ckpt_seed{s}_gen00300.npz") for s in range(20)]
    c += [dict(name=f"BTO-s{s}-b300", kind="bto", alpha=0.0,
               ckpt=f"results/es_bto/ckpt_seed{s}_gen00300.npz") for s in range(3)]
    return c


def main():
    ap = argparse.ArgumentParser(description="롤 동역학 진단")
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--out", default="results/r9_roll_diag.csv")
    a = ap.parse_args()
    env_mod.dyn_step = _rec_step
    rows = []
    for spec in conditions():
        pol = build_condition(spec)
        alpha = float(spec.get("alpha", 0.0))
        _REC.clear()
        for ov in (2, 3):
            opp = BTPolicy(version=ov)
            for i in range(a.n_seeds):
                env = DogfightEnv()
                orig_reset = env.reset

                def reset(*args, _o=orig_reset, _e=env, **kw):
                    out = _o(*args, **kw)
                    _BLUE["obj"] = _e.blue
                    return out
                env.reset = reset
                run_episode(pol, opp, seed=10_000 + i, alpha=alpha, alpha_red=0.0, env=env)
        p = np.degrees(np.abs(np.array([r[0] for r in _REC])))
        lw = np.array([r[1] for r in _REC])
        rows.append(dict(cond=spec["name"], n_steps=len(p),
                         p_median=float(np.median(p)), p99=float(np.percentile(p, 99)),
                         p_max=float(p.max()),
                         frac_over_90=float((p > 90).mean()), frac_over_180=float((p > 180).mean()),
                         frac_long_way=float(lw.mean())))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    fam = df["cond"].str.extract(r"^(BTO|BT|RL)")[0]
    print(df.groupby(fam)[["p_median", "p99", "p_max", "frac_over_90", "frac_over_180",
                           "frac_long_way"]].mean().to_string())


if __name__ == "__main__":
    main()
