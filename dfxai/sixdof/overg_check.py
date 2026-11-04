"""
6자유도 과G 위반 판정 점검 (C안 E3).

같은 교전에서 과G 위반율을 두 정의로 함께 센다.
  옛 정의 : 적분 스텝 끝의 순간 Nz > 8 G (2026-10-01 교차검증에서 쓴 것)
  새 정의 : min(0.05초 평균 Nz, 3자유도식 1차 지연 지령) > 8 G (env6dof.py 머리말)
3자유도 본실험(results/main, 같은 시드·상대·진영)의 위반율을 옆에 놓아, 새 정의가 3자유도와 같은
뜻이 되었는지 본다. 3자유도에서 8 G 를 넘지 않는 정책(BT-v1, BT-v3, Shield)은 새 정의에서도 0 이어야 한다.

    python -m dfxai.sixdof.overg_check --n-seeds 25 --workers 11
출력: results/sixdof/overg_check.csv, 화면에 표
"""
from __future__ import annotations
import argparse, os
import pandas as pd
from multiprocessing import Pool

from ..experiments.run_eval import build_condition
from ..agents.registry import make_bt, label
from .env6dof import SixDofEnv

CONDS = [dict(name="BT-v1", kind="bt", bt_version=1), dict(name="BT-v2", kind="bt", bt_version=2),
         dict(name="BT-v3", kind="bt", bt_version=3),
         dict(name="SHD-s11-b300", kind="shield", ckpt="results/es/ckpt_seed11_gen00300.npz", bt_version=3),
         dict(name="RL-s11-b300-a1.00", kind="rl", ckpt="results/es/ckpt_seed11_gen00300.npz"),
         dict(name="RL-s13-b300-a1.00", kind="rl", ckpt="results/es/ckpt_seed13_gen00300.npz")]


class _CountingEnv(SixDofEnv):
    """새 정의로 판정하면서 옛 정의(순간 Nz)도 함께 센다."""

    def reset(self, *a, **k):
        self.old = {}
        return super().reset(*a, **k)

    def _rule_load(self, state):
        key = id(state)
        self.old[key] = self.old.get(key, 0) + int(state.n > self.ec.over_g_limit)
        return super()._rule_load(state)


_ENV = None


def _job(args):
    global _ENV
    spec, ov, seed, side = args
    _ENV = _ENV or _CountingEnv()
    env = _ENV
    pol, opp = build_condition(spec), make_bt(ov)
    alpha = 1.0 if spec["kind"] in ("rl", "shield") else 0.0
    blue, red = (pol, opp) if side == 0 else (opp, pol)
    ob, orr = env.reset(seed=seed, alpha=alpha if side == 0 else 0.0, alpha_red=alpha if side == 1 else 0.0)
    for _ in range(env.max_steps + 1):
        ob, orr, done = env.step(blue.act(ob), red.act(orr))
        if done:
            break
    res = env.result()
    me = env.blue if side == 0 else env.red
    n_sub = max(1, res.steps * env.ec.n_substeps)
    new = (res.violations_blue if side == 0 else res.violations_red)["over_g"]
    return dict(cond=spec["name"], opponent=label(ov), seed=seed, side=side,
                over_g_old=env.old.get(id(me), 0) / n_sub, over_g_new=new)


def main():
    ap = argparse.ArgumentParser(description="6자유도 과G 판정 점검 (C안 E3)")
    ap.add_argument("--n-seeds", type=int, default=25)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results/sixdof/overg_check.csv")
    a = ap.parse_args()
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    jobs = [(c, ov, 10_000 + i, side) for c in CONDS for ov in (2, 3)
            for i in range(a.n_seeds) for side in (0, 1)]
    with Pool(a.workers) as p:
        df = pd.DataFrame(p.map(_job, jobs, chunksize=4))
    e3 = pd.read_csv("results/main/episodes.csv")
    e3 = e3[e3.cond.isin(df.cond.unique()) & (e3.seed < 10_000 + a.n_seeds)]
    df = df.merge(e3[["cond", "opponent", "seed", "side", "viol_over_g"]].rename(
        columns={"viol_over_g": "over_g_3dof"}), on=["cond", "opponent", "seed", "side"], how="left")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    t = df.groupby("cond", sort=False)[["over_g_3dof", "over_g_old", "over_g_new"]].mean()
    pd.set_option("display.width", 160)
    print(f"교전 {len(df)}회 (정책 {len(CONDS)} × 상대 BT-v2·BT-v3 × 시드 {a.n_seeds} × 진영 2)")
    print(t.round(4).to_string())


if __name__ == "__main__":
    main()
