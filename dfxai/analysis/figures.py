"""
논문 본문 그림 생성기 (report.py 가 만드는 기본 3종 외의 그림).

    python -m dfxai.analysis.figures --outdir results/main --esdir results/es

생성물 (results/<run>/paper_figs/):
  fig_learning_curve.png   ES 적합도 학습곡선 (학습 시드별)
  fig_budget_sweep.png     학습예산 x alpha 에 따른 성능 / D* / 위반율  (핵심 그림)
  fig_tradeoff_families.png 성능-설명가능성 평면을 (예산, alpha) 로 집계한 파레토 그림
  fig_traj_<cond>.png      대표 교전 궤적 (조건별 1장)

집계 규칙
---------
같은 (예산, alpha) 를 학습 시드가 여럿 공유하면 시드 평균 ± 표준편차로
그립니다. 학습 시드가 1개뿐이면 오차막대 없이 점만 찍습니다.
"""
from __future__ import annotations
import argparse, glob, json, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..env import DogfightEnv, run_episode
from ..experiments.run_eval import build_condition
from ..viz.plots import trajectory_figure

_PAT = re.compile(r"^(RL|HYB)-(?:s(\d+)-)?b(\d+)-a([0-9.]+)$")


def parse_cond(name: str) -> dict:
    """조건명 -> {family, train_seed, budget, alpha}. BT 는 budget=0, alpha=0."""
    if name.startswith("BT-v"):
        return dict(family="BT", train_seed=-1, budget=0, alpha=0.0,
                    bt_version=int(name[4:]))
    m = _PAT.match(name)
    if not m:
        return dict(family="?", train_seed=-1, budget=np.nan, alpha=np.nan)
    fam, s, b, a = m.groups()
    return dict(family="RL" if fam == "RL" else "Hybrid",
                train_seed=int(s) if s is not None else 0,
                budget=int(b), alpha=float(a), bt_version=-1)


def load_merged(outdir: str) -> pd.DataFrame:
    m = pd.read_csv(os.path.join(outdir, "merged.csv"))
    meta = pd.DataFrame([parse_cond(c) for c in m["cond"]])
    return pd.concat([m.reset_index(drop=True), meta], axis=1)


# ------------------------------------------------------------ 학습곡선
def plot_learning_curve(esdir: str, path: str):
    files = sorted(glob.glob(os.path.join(esdir, "history_*.json")))
    if not files:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for f in files:
        tag = os.path.basename(f)[len("history_"):-len(".json")]
        h = pd.DataFrame(json.load(open(f)))
        ax.plot(h["gen"], h["fit_mean"], lw=1.2, label=f"{tag} (pop. mean)")
        ax.plot(h["gen"], h["fit_max"], lw=0.8, ls="--", alpha=0.6,
                label=f"{tag} (pop. max)")
    ax.set_xlabel("Generation"); ax.set_ylabel("Fitness (shaped)")
    ax.set_title("Evolution-strategy learning curve")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)
    return path


# ------------------------------------------------------------ 예산 스윕
def _agg(d: pd.DataFrame, col: str) -> pd.DataFrame:
    g = d.groupby(["budget", "alpha"])[col]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out["std"] = out["std"].fillna(0.0)
    return out


def plot_budget_sweep(m: pd.DataFrame, path: str):
    learn = m[m["family"].isin(["RL", "Hybrid"])].dropna(subset=["budget"])
    bt = m[m["family"] == "BT"]
    if learn.empty:
        return None
    metrics = [("score", "Combat score"), ("d_star", r"$D^*$ (min depth @ 0.95)"),
               ("viol_total", "Violation rate / step")]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    alphas = sorted(learn["alpha"].unique())
    cmap = plt.get_cmap("viridis")
    for ax, (col, lab) in zip(axes, metrics):
        if col not in learn.columns:
            ax.set_visible(False); continue
        a = _agg(learn, col)
        for k, al in enumerate(alphas):
            s = a[a["alpha"] == al].sort_values("budget")
            ax.errorbar(s["budget"], s["mean"], yerr=s["std"] if s["count"].max() > 1 else None,
                        marker="o", ms=4, lw=1.2, capsize=2,
                        color=cmap(k / max(1, len(alphas) - 1)),
                        label=fr"$\alpha$={al:.2f}")
        for _, r in bt.iterrows():
            ax.axhline(r[col], color="0.4", ls=":", lw=0.9)
            ax.text(ax.get_xlim()[1], r[col], f" {r['cond']}", fontsize=6,
                    va="center", ha="left", color="0.3")
        ax.set_xlabel("Training budget (ES generations)"); ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, title="mix dial", title_fontsize=7)
    fig.suptitle("Effect of training budget and BT/RL mix (dotted = BT baselines)", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)
    return path


# ------------------------------------------------------------ 집계 파레토
def plot_tradeoff_families(m: pd.DataFrame, path: str):
    d = m.dropna(subset=["d_star"])
    if d.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    bt = d[d["family"] == "BT"]
    ax.scatter(bt["score"], bt["d_star"], marker="o", s=80, c="tab:blue",
               edgecolors="k", zorder=4, label="BT (rule)")
    for _, r in bt.iterrows():
        ax.annotate(r["cond"], (r["score"], r["d_star"]), fontsize=7,
                    xytext=(5, 4), textcoords="offset points")
    learn = d[d["family"].isin(["RL", "Hybrid"])]
    if not learn.empty:
        g = learn.groupby(["budget", "alpha"]).agg(
            score=("score", "mean"), score_sd=("score", "std"),
            d_star=("d_star", "mean"), d_sd=("d_star", "std"), n=("score", "size")).reset_index()
        cmap = plt.get_cmap("plasma")
        budgets = sorted(g["budget"].unique())
        for k, b in enumerate(budgets):
            s = g[g["budget"] == b].sort_values("alpha")
            col = cmap(0.15 + 0.7 * k / max(1, len(budgets) - 1))
            ax.errorbar(s["score"], s["d_star"],
                        xerr=s["score_sd"].fillna(0) if s["n"].max() > 1 else None,
                        yerr=s["d_sd"].fillna(0) if s["n"].max() > 1 else None,
                        fmt="-s", ms=5, lw=1.0, capsize=2, color=col,
                        label=f"budget {b} gen", zorder=3)
            for _, r in s.iterrows():
                ax.annotate(fr"$\alpha$={r['alpha']:.2f}", (r["score"], r["d_star"]),
                            fontsize=6, xytext=(4, -8), textcoords="offset points", color=col)
    # 전체 파레토 프론티어
    x, y = d["score"].values, d["d_star"].values
    idx = np.argsort(-x); best, keep = np.inf, []
    for i in idx:
        if y[i] < best:
            keep.append(i); best = y[i]
    keep = sorted(keep, key=lambda i: x[i])
    ax.plot(x[keep], y[keep], "k--", lw=1, alpha=0.6, label="Pareto front", zorder=2)
    ax.set_xlabel("Combat score (win=1, draw=0.5; mean over 2 opponents)")
    ax.set_ylabel(r"Explainability cost $D^*$ (min surrogate depth, fidelity 0.95)")
    ax.set_title("Performance–explainability plane (aggregated over training seeds)")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)
    return path


# ------------------------------------------------------------ 대표 궤적
def plot_representative_trajectories(outdir: str, figdir: str, seed: int = 10_000,
                                     opponent: int = 3, max_conds: int = 8):
    """manifest 의 조건 중 BT 3종 + 각 계열의 최종 예산 조건을 골라 궤적을 그립니다.

    같은 seed 를 쓰므로 초기조건이 동일하고, 조건 간 차이가 곧 정책 차이입니다.
    """
    man = json.load(open(os.path.join(outdir, "manifest.json"), encoding="utf-8"))
    conds = man.get("conditions")
    if not conds:                      # 합성 데이터 등 실제 정책이 없는 실행
        return []
    picked = [c for c in conds if c["kind"] == "bt"]
    learned = [c for c in conds if c["kind"] != "bt"]
    if learned:
        maxb = max(c.get("budget", 0) for c in learned)
        top = [c for c in learned if c.get("budget", 0) == maxb]
        seen = set()
        for c in sorted(top, key=lambda c: c["alpha"]):
            key = c["alpha"]
            if key not in seen:
                picked.append(c); seen.add(key)
    out = []
    for c in picked[:max_conds]:
        pol = build_condition(c)
        alpha = float(c.get("alpha", 1.0 if c["kind"] == "rl" else 0.0))
        env = DogfightEnv(record_traj=True)
        from ..agents.bt import BTPolicy
        res = run_episode(pol, BTPolicy(version=opponent), seed=seed, alpha=alpha,
                          alpha_red=0.0, env=env)
        who = {1: c["name"], -1: f"BT-v{opponent}", 0: "draw"}[res.winner]
        ttl = (f"{c['name']} (blue) vs BT-v{opponent} (red), seed {seed} — "
               f"{res.outcome}, winner: {who}, HP {res.hp_blue:.0f}:{res.hp_red:.0f}")
        safe = c["name"].replace("/", "_").replace(" ", "_")
        p = os.path.join(figdir, f"fig_traj_{safe}.png")
        trajectory_figure(np.asarray(env.traj), env.events, p, ttl,
                          blue_name=c["name"], red_name=f"BT-v{opponent}")
        out.append(p)
    return out


def make_figures(outdir: str, esdir: str = "results/es", traj_seed: int = 10_000,
                 traj_opponent: int = 3) -> list[str]:
    figdir = os.path.join(outdir, "paper_figs")
    os.makedirs(figdir, exist_ok=True)
    made = []
    p = plot_learning_curve(esdir, os.path.join(figdir, "fig_learning_curve.png"))
    if p: made.append(p)
    m = load_merged(outdir)
    p = plot_budget_sweep(m, os.path.join(figdir, "fig_budget_sweep.png"))
    if p: made.append(p)
    p = plot_tradeoff_families(m, os.path.join(figdir, "fig_tradeoff_families.png"))
    if p: made.append(p)
    made += plot_representative_trajectories(outdir, figdir, traj_seed, traj_opponent)
    return made


def main():
    ap = argparse.ArgumentParser(description="논문 그림 생성")
    ap.add_argument("--outdir", default="results/main")
    ap.add_argument("--esdir", default="results/es")
    ap.add_argument("--traj-seed", type=int, default=10_000)
    ap.add_argument("--traj-opponent", type=int, default=3)
    a = ap.parse_args()
    for p in make_figures(a.outdir, a.esdir, a.traj_seed, a.traj_opponent):
        print(p)


if __name__ == "__main__":
    main()
