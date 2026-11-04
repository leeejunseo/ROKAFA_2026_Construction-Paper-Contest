"""
C안 평가 파이프라인 (E8): 상대별 평가 → 상호 대전 → D* → 결과 문서 자동 생성.

    python -m dfxai.cplan.evaluate all --env 6dof --workers 11
    python -m dfxai.cplan.evaluate opp    --env 6dof --workers 11      상대별 평가 + D*
    python -m dfxai.cplan.evaluate league --env 6dof --workers 11      하이브리드 포함 상호 대전
    python -m dfxai.cplan.evaluate report --env 6dof                   paper/results_cplan_6dof.md

평가 대상 (모은 학습 결과에서 자동으로 만든다, plan.json 의 기준선 이름 사용)
  기준선 BT              BT-<기준선>
  초기화형               INIT-s<N>-b<G>          results/es6_init/ckpt_seed<N>_gen<G>.npz
  선형 혼합형 α          LIN-s<N>-b<G>-a0.25/0.50/0.75   (초기화형 정책 + 기준선 BT)
  감독형                 SHD-s<N>-b<G>           (초기화형 정책 + 기준선 BT 의 안전 노드)
  잔차형·게이팅형·파라미터 최적화형   RES- / GATE- / BTO-s<N>-b<G>
상대별 평가: 학습 상대(plan.train_opponents)와 보류 상대(plan.heldout_opponents) 각각 시드 100 × 진영 2.
  D* 는 진영 0 교전 중 상대마다 --record 개(기본 20)의 궤적으로 계산한다.
상호 대전: 유형(초기화·α 3종·감독·잔차·게이팅·최적화) 쌍마다 시드 N 대 시드 N+k (k = 1..--shifts) 를 붙이고
  기준선은 전원과 붙는다. 같은 신경망끼리(같은 시드) 붙지 않으며, 모두 학습 중에 서로 만난 적이 없다.
  레이팅은 Bradley–Terry (무승부 = 반승), 기준선 = 0 인 Elo 척도.
"""
from __future__ import annotations
import argparse, glob, json, math, os, re, sys
import numpy as np
import pandas as pd
from multiprocessing import Pool

from .job import ROOT, load_plan, outdir as train_dir
from ..agents.registry import make_bt, label, canon

ALPHAS = (0.25, 0.5, 0.75)
TYPES = ("INIT", "LIN-a0.25", "LIN-a0.50", "LIN-a0.75", "SHD", "RES", "GATE", "BTO")
TYPE_KO = {"BASE": "기준선 BT", "INIT": "초기화형", "LIN-a0.25": "선형 혼합 α=0.25", "LIN-a0.50": "선형 혼합 α=0.50",
           "LIN-a0.75": "선형 혼합 α=0.75", "SHD": "감독형", "RES": "잔차형", "GATE": "게이팅형", "BTO": "파라미터 최적화형"}


def eval_dir(env: str) -> str:
    return f"results/cplan/eval_{env}"


def ptype(name: str) -> str:
    if name.startswith("BT-"):
        return "BASE"
    if name.startswith("LIN"):
        return "LIN-" + name.rsplit("-", 1)[1]
    return name.split("-")[0]


def available_bts(names) -> list:
    out = []
    for n in names:
        try:
            make_bt(n)
            out.append(canon(n))
        except Exception as e:
            print(f"[건너뜀] 상대 {n}: {e}", flush=True)
    return out


def conditions(plan: dict, env: str, gen: int) -> list[dict]:
    base = plan["baseline"]
    conds = [dict(name=label(base), kind="bt", bt=base)]
    seeds = lambda k: sorted(int(re.search(r"seed(\d+)_", p).group(1))
                             for p in glob.glob(os.path.join(train_dir(plan, k, env), f"ckpt_seed*_gen{gen:05d}.npz")))
    ck = lambda k, s: os.path.join(train_dir(plan, k, env), f"ckpt_seed{s}_gen{gen:05d}.npz")
    for s in seeds("init"):
        c = dict(ckpt=ck("init", s), train_seed=s, budget=gen, bt=base)
        conds.append(dict(name=f"INIT-s{s}-b{gen}", kind="rl", alpha=1.0, **c))
        conds += [dict(name=f"LIN-s{s}-b{gen}-a{a:.2f}", kind="hybrid", alpha=a, **c) for a in ALPHAS]
        conds.append(dict(name=f"SHD-s{s}-b{gen}", kind="shield", alpha=1.0, **c))
    for k, pre, kind in (("residual", "RES", "residual"), ("gating", "GATE", "gating"), ("bto", "BTO", "bto")):
        for s in seeds(k):
            conds.append(dict(name=f"{pre}-s{s}-b{gen}", kind=kind, ckpt=ck(k, s), train_seed=s, budget=gen,
                              alpha=0.0 if kind == "bto" else 1.0))
    return conds


# ---------------------------------------------------------------- 상대별 평가
def run_opp(plan, env, gen, workers, n_seeds, record):
    from ..experiments.run_eval import run
    from ..analysis.report import compute_xai
    out = os.path.join(eval_dir(env), f"opp_b{gen}")
    conds = conditions(plan, env, gen)
    opps = available_bts(plan["train_opponents"] + plan["heldout_opponents"])
    print(f"상대별 평가: 정책 {len(conds)}개 × 상대 {[label(o) for o in opps]}", flush=True)
    run(conds, tuple(opps), n_seeds, workers, out, record_episodes=record,
        env_opts={"sixdof": True} if env == "6dof" else {})
    xai = compute_xai(out)
    xai.to_csv(os.path.join(out, "xai.csv"), index=False)
    return out


# ---------------------------------------------------------------- 상호 대전
def _league_job(args):
    from ..experiments.run_eval import build_condition, make_env
    from ..env import run_episode
    sa, sb, seed, opts = args
    env = make_env(opts)
    pa, pb = build_condition(sa), build_condition(sb)
    al = lambda s: float(s.get("alpha", 0.0))
    r1 = run_episode(pa, pb, seed=seed, alpha=al(sa), alpha_red=al(sb), env=env)
    pa, pb = build_condition(sa), build_condition(sb)
    r2 = run_episode(pb, pa, seed=seed, alpha=al(sb), alpha_red=al(sa), env=env)
    sc = lambda w: 1.0 if w == 1 else (0.5 if w == 0 else 0.0)
    return dict(a=sa["name"], b=sb["name"], seed=seed, s1=sc(r1.winner), s2=sc(-r2.winner),
                o1=r1.outcome, o2=r2.outcome)


def league_pairs(conds: list[dict], shifts: int) -> list[tuple]:
    by = {}
    base = None
    for c in conds:
        t = ptype(c["name"])
        if t == "BASE":
            base = c
        else:
            by.setdefault(t, {})[c["train_seed"]] = c
    pairs = []
    types = [t for t in TYPES if t in by]
    for i, t1 in enumerate(types):
        for t2 in types[i:]:
            s1 = sorted(by[t1]); s2 = sorted(by[t2])
            for j, s in enumerate(s1):
                for k in range(1, shifts + 1):
                    o = s2[(s2.index(s) + k) % len(s2)] if s in s2 else s2[(j + k) % len(s2)]
                    if o != s:
                        pairs.append((by[t1][s], by[t2][o]))
    if base:
        pairs += [(base, c) for t in types for c in by[t].values()]
    return pairs


def bradley_terry(df: pd.DataFrame, iters: int = 2000) -> pd.Series:
    """무승부 = 반승으로 센 Bradley–Terry 강도(MM 알고리즘). Elo 척도(400·log10)로 돌려준다."""
    names = sorted(set(df.a) | set(df.b))
    ix = {n: i for i, n in enumerate(names)}
    n = len(names)
    W = np.zeros((n, n))
    for r in df.itertuples():
        W[ix[r.a], ix[r.b]] += r.s1 + r.s2
        W[ix[r.b], ix[r.a]] += 2.0 - (r.s1 + r.s2)
    W += 0.5 * (W + W.T > 0)                       # 전승·전패 발산 방지용 아주 약한 사전값
    N = W + W.T
    p = np.ones(n)
    for _ in range(iters):
        denom = (N / (p[:, None] + p[None, :])).sum(axis=1)
        p_new = W.sum(axis=1) / np.maximum(denom, 1e-12)
        p_new /= np.exp(np.mean(np.log(p_new)))
        if np.max(np.abs(np.log(p_new) - np.log(p))) < 1e-10:
            p = p_new; break
        p = p_new
    return pd.Series(400.0 * np.log10(p), index=names)


def run_league(plan, env, gen, workers, n_seeds, shifts):
    out = os.path.join(eval_dir(env), f"league_b{gen}")
    os.makedirs(out, exist_ok=True)
    conds = conditions(plan, env, gen)
    pairs = league_pairs(conds, shifts)
    opts = {"sixdof": True} if env == "6dof" else {}
    jobs = [(a, b, 30_000 + s, opts) for a, b in pairs for s in range(n_seeds)]
    print(f"상호 대전: 짝 {len(pairs)}개 × 시드 {n_seeds} × 진영 2 = {2 * len(jobs)}교전", flush=True)
    with Pool(workers) as p:
        rows = p.map(_league_job, jobs, chunksize=4)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "league_episodes.csv"), index=False)
    g = df.groupby(["a", "b"])[["s1", "s2"]].mean().reset_index()
    rating = bradley_terry(df)
    base = label(plan["baseline"])
    if base in rating:
        rating -= rating[base]
    sc = pd.concat([pd.DataFrame(dict(cond=g.a, score=(g.s1 + g.s2) / 2)),
                    pd.DataFrame(dict(cond=g.b, score=1 - (g.s1 + g.s2) / 2))]).groupby("cond").score.mean()
    pd.DataFrame(dict(cond=rating.index, rating=rating.values,
                      league_score=sc.reindex(rating.index).values)).to_csv(os.path.join(out, "league_ratings.csv"),
                                                                          index=False)
    return out


# ---------------------------------------------------------------- 결과 문서
def per_policy(plan, env, gen) -> pd.DataFrame:
    d = os.path.join(eval_dir(env), f"opp_b{gen}")
    e = pd.read_csv(os.path.join(d, "episodes.csv"))
    train = {label(o) for o in plan["train_opponents"]}
    e["group"] = np.where(e.opponent.isin(train), "train", "heldout")
    e["kill_win"] = (e.win == 1) & (e.outcome == "gun_kill")
    e["time_win"] = (e.win == 1) & (e.outcome == "timeout")
    agg = e.groupby("cond").agg(score_all=("score", "mean"), kill_win=("kill_win", "mean"),
                                time_win=("time_win", "mean"), win=("win", "mean"), duration=("duration", "mean"),
                                damage=("damage_dealt", "mean"), viol_deck=("viol_deck", "mean"),
                                viol_over_g=("viol_over_g", "mean"))
    by = e.groupby(["cond", "group"]).score.mean().unstack()
    agg["score_train"] = by.get("train")
    agg["score_heldout"] = by.get("heldout")
    agg["gap"] = agg.score_train - agg.score_heldout
    agg["timeout_dependence"] = agg.time_win / agg.win.replace(0, np.nan)
    agg["dmg_per_min"] = agg.damage / (agg.duration / 60.0)
    for o in sorted(e.opponent.unique()):
        agg[f"score_vs_{o}"] = e[e.opponent == o].groupby("cond").score.mean()
    agg = agg.reset_index()
    x = pd.read_csv(os.path.join(d, "xai.csv"))[["cond", "d_star", "d_star_090", "fidelity_at_4"]]
    agg = agg.merge(x, on="cond", how="left")
    lr = os.path.join(eval_dir(env), f"league_b{gen}", "league_ratings.csv")
    if os.path.exists(lr):
        agg = agg.merge(pd.read_csv(lr), on="cond", how="left")
    agg["type"] = agg.cond.map(ptype)
    base = agg[agg.type == "BASE"].iloc[0]
    for c in ("score_train", "score_heldout", "d_star", "d_star_090"):
        agg[f"d_{c}"] = agg[c] - base[c]
    return agg


def report(plan, env, gen, out_md: str) -> str:
    from ..analysis.paper import md_table
    a = per_policy(plan, env, gen)
    a.to_csv(os.path.join(eval_dir(env), f"per_policy_b{gen}.csv"), index=False)
    cols = ["score_train", "score_heldout", "gap", "rating", "kill_win", "timeout_dependence", "dmg_per_min",
            "d_star", "d_star_090", "d_score_train", "d_score_heldout", "d_d_star"]
    cols = [c for c in cols if c in a]
    fam = a.groupby("type")[cols].agg(["mean", "std"])
    fam.columns = [f"{c}_{s}" for c, s in fam.columns]
    order = ["BASE"] + [t for t in TYPES if t in fam.index]
    rows = []
    for t in order:
        r = {"유형": TYPE_KO[t], "n": int((a.type == t).sum())}
        for c in cols:
            m, s = fam.loc[t, f"{c}_mean"], fam.loc[t, f"{c}_std"]
            r[c] = "–" if np.isnan(m) else (f"{m:.3f}" if (np.isnan(s) or r["n"] == 1) else f"{m:.3f} ± {s:.3f}")
        rows.append(r)
    t1 = pd.DataFrame(rows)
    opp_cols = [c for c in a.columns if c.startswith("score_vs_")]
    t2 = a.groupby("type")[opp_cols].mean().reindex(order).reset_index()
    t2["type"] = t2["type"].map(TYPE_KO)
    train = ", ".join(label(o) for o in plan["train_opponents"])
    held = ", ".join(label(o) for o in available_bts(plan["heldout_opponents"]))
    lines = [f"# C안 평가 결과 ({env.upper()}, {gen}세대, 자동 생성)", "",
             f"기준선 BT: {label(plan['baseline'])}. 학습 상대: {train}. 보류 상대: {held}.",
             "실행: `python -m dfxai.cplan.evaluate all --env " + env + "`. 사전 판정 기준: `paper/C안_사전판정기준.md`.", "",
             "열: score_train·score_heldout = 학습·보류 상대 평균 점수(승 1, 무 0.5), gap = 일반화 격차(학습 − 보류),",
             "rating = 상호 대전 Bradley–Terry 레이팅(기준선 = 0, Elo 척도), kill_win = 격추승 비율,",
             "timeout_dependence = 승리 중 시간종료 승의 비율, dmg_per_min = 분당 타격량, d_* = 기준선 대비 차이.", "",
             "## 1. 유형별 (평균 ± 시드 간 표준편차)", "", md_table(t1), "",
             "## 2. 상대별 점수", "", md_table(t2), "",
             "## 3. 정책별", "", f"`{eval_dir(env)}/per_policy_b{gen}.csv`", ""]
    text = "\n".join(lines)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(ROOT)
    ap = argparse.ArgumentParser(description="C안 평가 파이프라인 (E8)")
    ap.add_argument("cmd", choices=("conditions", "opp", "league", "report", "all"))
    ap.add_argument("--env", choices=("6dof", "3dof"), default="6dof")
    ap.add_argument("--gen", type=int, default=None, help="평가할 체크포인트 세대 (기본: plan.generations)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-seeds", type=int, default=100, help="상대별 평가 시드 수")
    ap.add_argument("--record", type=int, default=20, help="D* 용 궤적: 상대마다 진영 0 교전 수")
    ap.add_argument("--league-seeds", type=int, default=10)
    ap.add_argument("--shifts", type=int, default=2, help="상호 대전에서 유형 쌍마다 붙일 다른 시드 수")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    plan = load_plan()
    gen = a.gen or plan["generations"]
    if a.cmd == "conditions":
        for c in conditions(plan, a.env, gen):
            print(c["name"], c.get("ckpt", ""))
        return
    if a.cmd in ("opp", "all"):
        run_opp(plan, a.env, gen, a.workers, a.n_seeds, a.record)
    if a.cmd in ("league", "all"):
        run_league(plan, a.env, gen, a.workers, a.league_seeds, a.shifts)
    if a.cmd in ("report", "all"):
        print(report(plan, a.env, gen, a.out or os.path.join("paper", f"results_cplan_{a.env}.md")))


if __name__ == "__main__":
    main()
