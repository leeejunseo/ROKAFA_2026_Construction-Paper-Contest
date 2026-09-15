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
from .labels import T, set_language, language

_PAT = re.compile(r"^(RL|HYB)-(?:s(\d+)-)?b(\d+)-a([0-9.]+)$")
_PAT2 = re.compile(r"^(BTO|SHD|PPO)-s(\d+)-b(\d+)$")


def parse_cond(name: str) -> dict:
    """조건명 -> {family, train_seed, budget, alpha}. BT 는 budget=0, alpha=0.

    계열: BT(행동트리) / RL(순수 학습) / Hybrid(선형 혼합) /
          BTO(상수 최적화 BT) / Shield(감독형 혼합).
    """
    if name.startswith("BT-v"):
        return dict(family="BT", train_seed=-1, budget=0, alpha=0.0,
                    bt_version=int(name[4:]))
    m2 = _PAT2.match(name)
    if m2:
        fam, s, b = m2.groups()
        family = {"BTO": "BTO", "SHD": "Shield", "PPO": "PPO"}[fam]
        return dict(family=family, train_seed=int(s), budget=int(b),
                    alpha=0.0 if fam == "BTO" else 1.0, bt_version=-1)
    m = _PAT.match(name)
    if not m:
        return dict(family="?", train_seed=-1, budget=np.nan, alpha=np.nan)
    fam, s, b, a = m.groups()
    return dict(family="RL" if fam == "RL" else "Hybrid",
                train_seed=int(s) if s is not None else 0,
                budget=int(b), alpha=float(a), bt_version=-1)


def read_json(path: str) -> dict:
    """manifest 는 과거 실행에서 시스템 코드페이지(cp949)로 저장됐을 수 있습니다."""
    for enc in ("utf-8", "cp949"):
        try:
            with open(path, encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    with open(path, encoding="utf-8", errors="replace") as f:
        return json.load(f)


def load_merged(outdir: str) -> pd.DataFrame:
    m = pd.read_csv(os.path.join(outdir, "merged.csv"))
    meta = pd.DataFrame([parse_cond(c) for c in m["cond"]])
    # report.py 의 family 열은 parse_cond 가 다시 만들므로 중복을 피해 버립니다.
    m = m.drop(columns=["family"], errors="ignore").reset_index(drop=True)
    return pd.concat([m, meta], axis=1)


# ------------------------------------------------------------ 학습곡선
def plot_learning_curve(esdir: str, path: str):
    files = sorted(glob.glob(os.path.join(esdir, "history_*.json")))
    if not files:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    win = 15
    for f in files:
        tag = os.path.basename(f)[len("history_"):-len(".json")]
        h = pd.DataFrame(read_json(f))
        # 세대마다 시드가 바뀌어 원값은 잡음이 큽니다. 이동평균(15세대)으로 그리고
        # 원값은 옅게 깔아 둡니다. 수용 검사가 켜진 실행이면 현재 θ 의 적합도를 씁니다.
        col = "fit_theta" if "fit_theta" in h.columns else "fit_mean"
        y = h[col].astype(float)
        line, = ax.plot(h["gen"], y.rolling(win, min_periods=1, center=True).mean(),
                        lw=1.6, label=f"{tag}")
        ax.plot(h["gen"], y, lw=0.5, alpha=0.18, color=line.get_color())
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel(T("Generation", "세대"))
    ax.set_ylabel(T("Fitness of current policy (shaped)", "현재 정책의 적합도 (셰이핑 포함)"))
    ax.set_title(T(f"ES learning curve ({win}-generation moving average; faint = raw)",
                   f"ES 학습곡선 ({win}세대 이동평균; 옅은 선 = 원값)"), fontsize=9)
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
    # D* 는 절단(17)과 시드 분산이 커서 F(4) 를 나란히 둡니다 — 방향이 같아야 주장이 섭니다.
    metrics = [("score", T("Combat score", "전투 점수")),
               ("d_star", T(r"$D^*$ (min depth @ 0.95)", r"$D^*$ (충실도 0.95 최소 깊이)")),
               ("fidelity_at_4", T(r"$F(4)$  (depth-4 fidelity, higher = simpler)", r"$F(4)$  (깊이 4 충실도, 높을수록 단순)")),
               ("viol_total", T("Violation rate / step", "스텝당 위반율"))]
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 3.8))
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
        ax.set_xlabel(T("Training budget (ES generations)", "학습 예산 (ES 세대)")); ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, title=T("mix dial", "혼합계수"), title_fontsize=7)
    fig.suptitle(T("Effect of training budget and BT/RL mix (dotted = BT baselines)",
                   "학습 예산과 BT/RL 혼합의 효과 (점선 = BT 기준선)"), fontsize=10)
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
               edgecolors="k", zorder=4, label=T("BT (rule)", "BT (규칙)"))
    for _, r in bt.iterrows():
        ax.annotate(r["cond"], (r["score"], r["d_star"]), fontsize=7,
                    xytext=(5, 4), textcoords="offset points")
    # 확장 계열: 최대 예산 조건만 시드 집계 (평균 ± 표준편차)
    ext_style = {"BTO": ("D", "tab:cyan", T("optimized BT (BTO)", "상수 최적화 BT (BTO)")),
                 "Shield": ("P", "tab:olive", T("shield hybrid (SHD)", "감독형 혼합 (SHD)")),
                 "PPO": ("X", "tab:gray", T("PPO from scratch", "밑바닥 PPO"))}
    for fam, (mk, col, lab) in ext_style.items():
        e = d[d["family"] == fam]
        if e.empty:
            continue
        e = e[e["budget"] == e["budget"].max()]
        ax.errorbar(e["score"].mean(), e["d_star"].mean(),
                    xerr=e["score"].std() if len(e) > 1 else None,
                    yerr=e["d_star"].std() if len(e) > 1 else None,
                    fmt=mk, ms=8, color=col, mec="k", capsize=2, label=lab, zorder=4)
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
                        label=T(f"budget {b} gen", f"예산 {b}세대"), zorder=3)
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
    ax.plot(x[keep], y[keep], "k--", lw=1, alpha=0.6, label=T("Pareto front", "파레토 프론티어"), zorder=2)
    ax.set_xlabel(T("Combat score (win=1, draw=0.5; mean over 2 opponents)", "전투 점수 (승 1, 무 0.5; 상대 2종 평균)"))
    ax.set_ylabel(T(r"Explainability cost $D^*$ (min surrogate depth, fidelity 0.95)", r"설명 비용 $D^*$ (대리모델 최소 깊이, 충실도 0.95)"))
    ax.set_title(T("Performance–explainability plane (aggregated over training seeds)", "성능–설명가능성 평면 (학습 시드 집계)"))
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)
    return path


# ------------------------------------------------------------ 대표 궤적
def pick_representative(conds: list[dict]) -> list[dict]:
    """BT 3종 + 최대 예산의 학습 조건을 alpha 별로 하나씩(학습 시드 0 우선).

    합성 데이터처럼 실제 정책이 없는 실행이면 빈 목록을 돌려줍니다.
    """
    picked = [c for c in conds if c["kind"] == "bt"]
    # 순수 학습·혼합 계열만 (PPO 는 kind 가 rl 이지만 이름으로 구분)
    learned = [c for c in conds if c["kind"] in ("rl", "hybrid")
               and c["name"].startswith(("RL-", "HYB-"))]
    if learned:
        maxb = max(c.get("budget", 0) for c in learned)
        top = sorted([c for c in learned if c.get("budget", 0) == maxb],
                     key=lambda c: (c["alpha"], c.get("train_seed", 0)))
        seen = set()
        for c in top:
            if c["alpha"] not in seen:
                picked.append(c); seen.add(c["alpha"])
    # 확장 계열은 최대 예산·학습 시드 0 하나씩
    for prefix in ("BTO-", "SHD-", "PPO-"):
        ext = [c for c in conds if c["name"].startswith(prefix)]
        if ext:
            maxb = max(c.get("budget", 0) for c in ext)
            picked.append(sorted([c for c in ext if c.get("budget", 0) == maxb],
                                 key=lambda c: c.get("train_seed", 0))[0])
    return picked


def choose_representative_seed(conds: list[dict], opponent: int = 3,
                               seed0: int = 10_000, n_try: int = 40) -> int:
    """모든 대표 조건에서 교전이 충분히 오래 이어지는 시드를 고릅니다.

    정면 조우 6초 만에 충돌로 끝나는 시드는 그림으로 쓸모가 없습니다.
    후보 시드마다 조건별 교전 길이의 최솟값을 구해 그 값이 가장 큰 시드를
    택합니다(같은 시드 = 같은 초기조건이므로 조건 간 비교가 공정합니다).
    """
    from ..agents.bt import BTPolicy
    pols = [(build_condition(c), float(c.get("alpha", 1.0 if c["kind"] == "rl" else 0.0)))
            for c in conds]
    best, best_seed = -1.0, seed0
    for s in range(seed0, seed0 + n_try):
        worst = np.inf
        for pol, alpha in pols:
            res = run_episode(pol, BTPolicy(version=opponent), seed=s, alpha=alpha)
            d = res.duration if res.outcome != "collision" else 0.0
            worst = min(worst, d)
            if worst <= best:
                break
        if worst > best:
            best, best_seed = worst, s
    return best_seed


def plot_representative_trajectories(outdir: str, figdir: str, seed: int | None = None,
                                     opponent: int = 3, max_conds: int = 8):
    """manifest 의 조건 중 BT 3종 + 각 계열의 최종 예산 조건을 골라 궤적을 그립니다.

    같은 seed 를 쓰므로 초기조건이 동일하고, 조건 간 차이가 곧 정책 차이입니다.
    seed=None 이면 choose_representative_seed() 로 자동 선택합니다.
    """
    man = read_json(os.path.join(outdir, "manifest.json"))
    picked = pick_representative(man.get("conditions") or [])[:max_conds]
    if not picked:
        return []
    if seed is None:
        seed = choose_representative_seed(picked, opponent)
        with open(os.path.join(figdir, "representative_seed.txt"), "w") as f:
            f.write(f"{seed}\n")
    out = []
    for c in picked:
        pol = build_condition(c)
        alpha = float(c.get("alpha", 1.0 if c["kind"] == "rl" else 0.0))
        env = DogfightEnv(record_traj=True)
        from ..agents.bt import BTPolicy
        res = run_episode(pol, BTPolicy(version=opponent), seed=seed, alpha=alpha,
                          alpha_red=0.0, env=env)
        who = {1: c["name"], -1: f"BT-v{opponent}", 0: T("draw", "무승부")}[res.winner]
        ttl = T(f"{c['name']} (blue) vs BT-v{opponent} (red), seed {seed} — "
                f"{res.outcome}, winner: {who}, HP {res.hp_blue:.0f}:{res.hp_red:.0f}",
                f"{c['name']} (청) 대 BT-v{opponent} (홍), 시드 {seed} — "
                f"{res.outcome}, 승자: {who}, HP {res.hp_blue:.0f}:{res.hp_red:.0f}")
        safe = c["name"].replace("/", "_").replace(" ", "_")
        p = os.path.join(figdir, f"fig_traj_{safe}.png")
        trajectory_figure(np.asarray(env.traj), env.events, p, ttl,
                          blue_name=c["name"], red_name=f"BT-v{opponent}")
        out.append(p)
    return out


def plot_trajectory_pair(outdir: str, figdir: str, seed: int, opponent: int = 3,
                         left: str = "BT-v2", right_kind: str = "rl") -> str | None:
    """논문 그림 2: 규칙 정책과 학습 정책의 대표 교전을 한 장에 나란히.

    왼쪽 BT-v2, 오른쪽 최대 예산의 순수 학습 정책(학습 시드 0). 같은 seed 이므로
    초기조건이 같고, 두 궤적의 차이가 곧 정책 차이입니다. 위 행은 평면 궤적,
    아래 행은 거리·ATA 시계열.
    """
    from ..agents.bt import BTPolicy
    from ..viz.plots import _ata_series, _fire_segments, BLUE, RED
    from ..config import EngagementConfig
    man = read_json(os.path.join(outdir, "manifest.json"))
    conds = man.get("conditions") or []
    lc = next((c for c in conds if c["name"] == left), None)
    learned = [c for c in conds if c["kind"] == right_kind]
    if not lc or not learned:
        return None
    maxb = max(c.get("budget", 0) for c in learned)
    rc = sorted([c for c in learned if c.get("budget", 0) == maxb],
                key=lambda c: c.get("train_seed", 0))[0]
    ec = EngagementConfig()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), gridspec_kw=dict(height_ratios=[1.5, 1]))
    for j, c in enumerate((lc, rc)):
        env = DogfightEnv(record_traj=True)
        alpha = float(c.get("alpha", 1.0 if c["kind"] == "rl" else 0.0))
        res = run_episode(build_condition(c), BTPolicy(version=opponent), seed=seed,
                          alpha=alpha, alpha_red=0.0, env=env)
        tr = np.asarray(env.traj); t = tr[:, 0]
        ax = axes[0, j]
        for x, y, fl, col, nm in ((tr[:, 1], tr[:, 2], tr[:, 22], BLUE, c["name"]),
                                  (tr[:, 11], tr[:, 12], tr[:, 23], RED, f"BT-v{opponent}")):
            ax.plot(y / 1000, x / 1000, color=col, lw=1.1, alpha=0.85, label=nm)
            for s, e in _fire_segments(t, fl):
                ax.plot(y[s:e] / 1000, x[s:e] / 1000, color=col, lw=3.2)
            ax.plot(y[0] / 1000, x[0] / 1000, "o", color=col, ms=6, mec="k", mew=0.6)
            ax.plot(y[-1] / 1000, x[-1] / 1000, "x", color=col, ms=8, mew=2)
        ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=0.3)
        ax.set_xlabel(T("East [km]", "동 [km]")); ax.set_ylabel(T("North [km]", "북 [km]"))
        ax.legend(fontsize=8, loc="best")
        who = {1: c["name"], -1: f"BT-v{opponent}", 0: T("draw", "무승부")}[res.winner]
        ax.set_title(T(f"({'ab'[j]}) {c['name']} vs BT-v{opponent} — {res.outcome}, winner {who}, "
                       f"HP {res.hp_blue:.0f}:{res.hp_red:.0f}",
                       f"({'ab'[j]}) {c['name']} 대 BT-v{opponent} — {res.outcome}, 승자 {who}, "
                       f"HP {res.hp_blue:.0f}:{res.hp_red:.0f}"), fontsize=9)
        ax = axes[1, j]
        ax.axhspan(ec.gun_range_min / 1000, ec.gun_range_max / 1000, color="0.85", alpha=0.6, lw=0,
                   label=T("gun range", "기총 사거리"))
        ax.plot(t, tr[:, 21] / 1000, color="k", lw=1.2, label=T("range", "거리"))
        ax.set_ylim(0, max(2.0, tr[:, 21].max() / 1000 * 1.05))
        ax.set_xlabel(T("Time [s]", "시간 [s]")); ax.set_ylabel(T("Range [km]", "거리 [km]"))
        ax2 = ax.twinx()
        ax2.plot(t, _ata_series(tr), color=BLUE, lw=1.0, alpha=0.8, label=T("ATA (blue)", "ATA (청)"))
        ax2.set_ylim(0, 180); ax2.set_ylabel(T("ATA [deg]", "ATA [도]"), color=BLUE)
        ax2.tick_params(axis="y", colors=BLUE)
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right", ncol=3); ax.grid(alpha=0.3)
    fig.suptitle(T(f"Rule policy vs learned policy on the same initial condition (seed {seed})",
                   f"같은 초기조건(시드 {seed})에서 규칙 정책과 학습 정책"), fontsize=10)
    fig.tight_layout()
    p = os.path.join(figdir, "fig_traj_pair.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    return p


def make_figures(outdir: str, esdir: str = "results/es", traj_seed: int | None = None,
                 traj_opponent: int = 3, lang: str = "en") -> list[str]:
    set_language(lang)
    figdir = os.path.join(outdir, "paper_figs" if language() == "en" else "paper_figs_ko")
    os.makedirs(figdir, exist_ok=True)
    made = []
    from .overview import draw as draw_overview
    man = read_json(os.path.join(outdir, "manifest.json"))
    conds = man.get("conditions") or []
    n_ts = len({c.get("train_seed", -1) for c in conds if c["kind"] != "bt"} - {-1}) or 1
    n_fights = len(conds) * len(man.get("opponents", [2, 3])) * int(man.get("n_seeds", 100)) * 2
    made.append(draw_overview(os.path.join(figdir, "fig_overview.png"), len(conds), n_fights, n_ts))
    p = plot_learning_curve(esdir, os.path.join(figdir, "fig_learning_curve.png"))
    if p: made.append(p)
    m = load_merged(outdir)
    p = plot_budget_sweep(m, os.path.join(figdir, "fig_budget_sweep.png"))
    if p: made.append(p)
    p = plot_tradeoff_families(m, os.path.join(figdir, "fig_tradeoff_families.png"))
    if p: made.append(p)
    made += plot_representative_trajectories(outdir, figdir, traj_seed, traj_opponent)
    rep = os.path.join(figdir, "representative_seed.txt")
    seed = traj_seed if traj_seed is not None else (int(open(rep).read()) if os.path.exists(rep) else 10_000)
    p = plot_trajectory_pair(outdir, figdir, seed, traj_opponent)
    if p: made.append(p)
    return made


def main():
    ap = argparse.ArgumentParser(description="논문 그림 생성")
    ap.add_argument("--outdir", default="results/main")
    ap.add_argument("--esdir", default="results/es")
    ap.add_argument("--traj-seed", type=int, default=None,
                    help="대표 궤적 시드. 생략하면 자동 선택 (paper_figs/representative_seed.txt 에 기록)")
    ap.add_argument("--traj-opponent", type=int, default=3)
    ap.add_argument("--lang", default="en", choices=("en", "ko"),
                    help="그림 라벨 언어. ko 는 paper_figs_ko/ 에 저장")
    a = ap.parse_args()
    for p in make_figures(a.outdir, a.esdir, a.traj_seed, a.traj_opponent, a.lang):
        print(p)


if __name__ == "__main__":
    main()
