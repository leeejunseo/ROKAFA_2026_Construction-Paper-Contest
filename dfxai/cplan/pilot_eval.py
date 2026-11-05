"""
6DOF 시범 학습 판정 (C안 E5 → G2): 학습이 행동 복제 출발점보다 점수를 올리는가.

    python -m dfxai.cplan.pilot_eval --dir results/cplan/pilots/6dof_btm_c612_seed0 --bc results/cplan/bc/bc_btm_6dof.npz --workers 11

출발점(행동 복제)과 체크포인트마다 순수 학습 정책(α = 1)을 같은 시드·진영 교대로 평가한다. 시드 40000부터라 학습·본평가 시드와
겹치지 않는다. 출발점 대비 차이는 시드 단위로 짝지어 부트스트랩 95% 구간을 낸다. 결과: <dir>/pilot_eval.md
상대(--opponents)와 환경 규칙은 기본으로 plan.json 을 따른다(학습 상대 + 보류 상대, 6자유도 기총 원추).
"""
from __future__ import annotations
import argparse, glob, os, re, sys
import numpy as np
import pandas as pd

from .job import load_plan
from ..agents.registry import label
from ..experiments.run_eval import run
from ..rl.es_runtime import plan_env_opts


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="6DOF 시범 학습 판정 (E5)")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--bc", required=True, help="출발점 행동 복제 체크포인트")
    ap.add_argument("--opponents", nargs="+", default=None, help="기본: plan.json 의 학습 상대 + 보류 상대")
    ap.add_argument("--n-seeds", type=int, default=50)
    ap.add_argument("--env", choices=("6dof", "3dof"), default="6dof")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    plan = load_plan()
    if a.opponents is None:
        a.opponents = plan["train_opponents"] + plan["heldout_opponents"]
    opts = plan_env_opts(plan, a.env)
    cks = sorted(glob.glob(os.path.join(a.dir, "ckpt_seed*_gen*.npz")))
    conds = [dict(name="gen000-BC", kind="rl", alpha=1.0, ckpt=a.bc)]
    gen_of = lambda p: int(re.search(r"gen(\d+)", p).group(1))
    conds += [dict(name=f"gen{gen_of(c):03d}", kind="rl", alpha=1.0, ckpt=c) for c in cks]
    out = os.path.join(a.dir, "pilot_eval")
    opps = tuple(int(o) if o.isdigit() else o for o in a.opponents)
    df = run(conds, opps, a.n_seeds, a.workers, out, record_episodes=0, seed0=40_000, env_opts=opts)
    per_seed = df.groupby(["cond", "opponent", "seed"]).score.mean().unstack(["opponent", "seed"])
    tab = df.groupby(["cond", "opponent"]).score.mean().unstack()
    tab["평균"] = df.groupby("cond").score.mean()
    is_train = df.opponent.isin({label(o) for o in plan["train_opponents"]})
    tab["학습상대"] = df[is_train].groupby("cond").score.mean()
    tab["보류상대"] = df[~is_train].groupby("cond").score.mean()
    df["hp_diff"] = df.damage_dealt - df.damage_taken
    tab["HP차이"] = df.groupby("cond").hp_diff.mean()
    tab["WEZ초"] = df.groupby("cond").wez_time.mean()
    tab["하드덱"] = df.groupby("cond").viol_deck.mean()
    rng = np.random.default_rng(0)
    base = per_seed.loc["gen000-BC"].values
    rows = []
    for c in per_seed.index:
        d = per_seed.loc[c].values - base
        boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
        rows.append(dict(cond=c, 차이=d.mean(), 구간_하한=np.quantile(boots, 0.025), 구간_상한=np.quantile(boots, 0.975)))
    diff = pd.DataFrame(rows).set_index("cond")
    res = tab.join(diff)
    last = res.index[-1]
    ok = res.loc[last, "구간_하한"] > 0
    from ..analysis.paper import md_table
    text = "\n".join([
        f"# 6DOF 시범 학습 판정 ({os.path.basename(a.dir)})", "",
        f"출발점: `{a.bc}`. 상대 {', '.join(label(o) for o in opps)} × 시드 {a.n_seeds}(40000부터) × 진영 교대. 점수: 승 1, 무 0.5. 환경 규칙: {opts}.",
        "차이 = 출발점 대비 평균 점수 차이(시드 단위 짝, 부트스트랩 95% 구간).", "",
        md_table(res.reset_index()), "",
        f"G2 판정(마지막 체크포인트 {last} 가 출발점보다 높고 구간 하한 > 0): **{'통과' if ok else '미달'}**", ""])
    with open(os.path.join(a.dir, "pilot_eval.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
