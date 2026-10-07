"""
논문용 3차원 교전 궤적 그림(한글 라벨).

    python -m dfxai.viz.paper_traj3d --outdir results/paper_figs_rewrite

<그림 3-2> 모의 환경에서의 교전 장면: BT-v2(청) 대 BT-v3(홍), 초기조건 10055.
<그림 4-3> 두 전술의 교전 궤적: (a) 치고 빠지기 — 학습 시드 11 대 BT-v2, 초기조건 10054
                               (b) 추격 격추 — 학습 시드 1 대 BT-v2, 초기조건 10016

교전은 본실험과 같은 난수원으로 다시 실행하므로 results/main/episodes.csv 의 해당 교전과 결과가 같다.
"""
from __future__ import annotations
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ..env import DogfightEnv
from ..agents.bt import BTPolicy
from ..agents.hybrid import MLPPolicy

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

BLUE, RED, FIRE = "#1f5fbf", "#c8402f", "#f2a900"
OUTCOME_KO = {"timeout": "시간종료", "gun_kill": "격추", "crash": "지면 충돌", "collision": "공중 충돌"}


def run(blue, red, seed: int, alpha: float):
    env = DogfightEnv(record_traj=True, traj_stride=2)
    ob, orr = env.reset(seed=seed, alpha=alpha, alpha_red=0.0)
    for _ in range(env.max_steps + 1):
        ob, orr, done = env.step(blue.act(ob), red.act(orr))
        if done:
            break
    T = np.asarray(env.traj)
    return T, env.result()


def draw(ax, T, res, blue_name: str, red_name: str, title: str):
    # 열: t, bx,by,bh,bv,bpsi,bgam,bmu,bn,bthr,bhp, rx,ry,rh,..., r, fire_b, fire_r
    bx, by, bh = T[:, 1] / 1000, T[:, 2] / 1000, T[:, 3] / 1000
    rx, ry, rh = T[:, 11] / 1000, T[:, 12] / 1000, T[:, 13] / 1000
    fb, fr = T[:, 22] > 0.5, T[:, 23] > 0.5
    ax.plot(by, bx, bh, color=BLUE, lw=1.6, label=blue_name)
    ax.plot(ry, rx, rh, color=RED, lw=1.6, label=red_name)
    # 사격 구간(WEZ 안)
    for x, y, h, f, lab in ((bx, by, bh, fb, "WEZ 안(사격 중)"), (rx, ry, rh, fr, None)):
        if f.any():
            ax.scatter(y[f], x[f], h[f], color=FIRE, s=7, depthshade=False, zorder=5, label=lab)
    # 시작·끝
    ax.scatter(by[0], bx[0], bh[0], color=BLUE, marker="o", s=40, edgecolor="k", zorder=6)
    ax.scatter(ry[0], rx[0], rh[0], color=RED, marker="o", s=40, edgecolor="k", zorder=6)
    end_b = "X" if res.hp_blue <= 0 else "s"
    end_r = "X" if res.hp_red <= 0 else "s"
    ax.scatter(by[-1], bx[-1], bh[-1], color=BLUE, marker=end_b, s=70, edgecolor="k", zorder=6)
    ax.scatter(ry[-1], rx[-1], rh[-1], color=RED, marker=end_r, s=70, edgecolor="k", zorder=6)
    # 하드덱(1,000 m) 면
    xs = np.concatenate([by, ry]); ys = np.concatenate([bx, rx]); hs = np.concatenate([bh, rh])
    pad = 0.5
    x0, x1, y0, y1 = xs.min() - pad, xs.max() + pad, ys.min() - pad, ys.max() + pad
    deck = Poly3DCollection([[(x0, y0, 1.0), (x1, y0, 1.0), (x1, y1, 1.0), (x0, y1, 1.0)]],
                            facecolor="#7a7a7a", alpha=0.12, edgecolor="#7a7a7a", lw=0.5)
    ax.add_collection3d(deck)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_zlim(0.0, max(hs.max() + 0.3, 2.0))
    ax.set_xlabel("동쪽 (km)", labelpad=4); ax.set_ylabel("북쪽 (km)", labelpad=4); ax.set_zlabel("고도 (km)", labelpad=2)
    ax.tick_params(labelsize=7.5, pad=1)
    ax.view_init(elev=24, azim=-58)
    ax.set_title(title, fontsize=10, pad=4)


def caption_result(res) -> str:
    who = {1: "청군 승", -1: "홍군 승", 0: "무승부"}[res.winner]
    return (f"{OUTCOME_KO.get(res.outcome, res.outcome)} · {res.duration:.0f}초 · {who} · "
            f"남은 체력 청 {res.hp_blue:.0f} / 홍 {res.hp_red:.0f}")


def legend_handles():
    from matplotlib.lines import Line2D
    return [Line2D([], [], color="k", marker="o", ls="", ms=6, label="시작 위치"),
            Line2D([], [], color="k", marker="s", ls="", ms=6, label="종료 위치"),
            Line2D([], [], color="k", marker="X", ls="", ms=7, label="격추된 기체"),
            Line2D([], [], color=FIRE, marker="o", ls="", ms=5, label="WEZ 안(사격 중)"),
            plt.Rectangle((0, 0), 1, 1, fc="#7a7a7a", alpha=0.25, label="하드덱(1,000 m)")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results/paper_figs_rewrite")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    # <그림 3-2>
    T, res = run(BTPolicy(2), BTPolicy(3), seed=10055, alpha=0.0)
    fig = plt.figure(figsize=(7.2, 5.0), dpi=200)
    ax = fig.add_subplot(111, projection="3d")
    draw(ax, T, res, "청군: BT-v2", "홍군: BT-v3", "")   # 제목은 논문 캡션에 둔다
    h1, l1 = ax.get_legend_handles_labels()
    keep = [(h, l) for h, l in zip(h1, l1) if l in ("청군: BT-v2", "홍군: BT-v3")]
    fig.legend([h for h, _ in keep] + legend_handles(), [l for _, l in keep] + [h.get_label() for h in legend_handles()],
               loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.12)
    p1 = os.path.join(a.outdir, "fig3_2_engagement.png")
    fig.savefig(p1, dpi=200); plt.close(fig)
    print(p1, caption_result(res))

    # <그림 4-3>
    rl11 = MLPPolicy.load("results/es/ckpt_seed11_gen00300.npz")
    rl1 = MLPPolicy.load("results/es/ckpt_seed1_gen00300.npz")
    Ta, ra = run(rl11, BTPolicy(2), seed=10054, alpha=1.0)
    Tb, rb = run(rl1, BTPolicy(2), seed=10016, alpha=1.0)
    fig = plt.figure(figsize=(11.0, 5.0), dpi=200)
    axa = fig.add_subplot(121, projection="3d")
    axb = fig.add_subplot(122, projection="3d")
    draw(axa, Ta, ra, "청군: 학습 기반(시드 11)", "홍군: BT-v2", "(a) 치고 빠지기\n" + caption_result(ra))
    draw(axb, Tb, rb, "청군: 학습 기반(시드 1)", "홍군: BT-v2", "(b) 추격 격추\n" + caption_result(rb))
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=BLUE, lw=2, label="청군: 학습 기반"), Line2D([], [], color=RED, lw=2, label="홍군: BT-v2")] + legend_handles()
    fig.legend(hs, [h.get_label() for h in hs], loc="lower center", ncol=7, fontsize=8, frameon=False)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.9, bottom=0.1, wspace=0.02)
    p2 = os.path.join(a.outdir, "fig4_3_tactics.png")
    fig.savefig(p2, dpi=200); plt.close(fig)
    print(p2, caption_result(ra), "|", caption_result(rb))


if __name__ == "__main__":
    main()
