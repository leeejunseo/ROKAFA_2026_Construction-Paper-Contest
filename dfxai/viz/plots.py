"""
논문용 정적 교전 그림 (matplotlib).

Tacview 재생은 심사 시연용이고, 논문 본문에는 정지 그림이 필요합니다.
`trajectory_figure()` 는 교전 1회를 3단 패널로 그립니다.

  (a) 평면 궤적   : 청/홍 비행경로, 시작점(○)·종료점(×), 사격 구간은 굵게
  (b) 고도 이력   : 하드덱(1,000 m) 점선 포함
  (c) 교전기하    : 상대거리(기총 사거리 띠 표시)와 청군 ATA(명중 원추 표시)

입력은 env.traj (24열, env.TRAJ_COLUMNS 순서) 와 env.events 입니다.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..config import EngagementConfig
from ..env import TRAJ_COLUMNS
from ..analysis.labels import T

_C = {name: i for i, name in enumerate(TRAJ_COLUMNS)}
BLUE, RED = "#1f5fbf", "#c8311f"


def _ata_series(traj: np.ndarray) -> np.ndarray:
    """청군 보어사이트 이탈각(도). 기록에는 없으므로 속도벡터로 재계산합니다."""
    bx, by, bh = traj[:, _C["bx"]], traj[:, _C["by"]], traj[:, _C["bh"]]
    rx, ry, rh = traj[:, _C["rx"]], traj[:, _C["ry"]], traj[:, _C["rh"]]
    psi, gam = traj[:, _C["bpsi"]], traj[:, _C["bgamma"]]
    los = np.stack([rx - bx, ry - by, rh - bh], axis=1)
    los /= np.maximum(np.linalg.norm(los, axis=1, keepdims=True), 1e-6)
    u = np.stack([np.cos(gam) * np.cos(psi), np.cos(gam) * np.sin(psi),
                  np.sin(gam)], axis=1)
    return np.degrees(np.arccos(np.clip((u * los).sum(axis=1), -1, 1)))


def _fire_segments(t: np.ndarray, flag: np.ndarray) -> list[tuple[int, int]]:
    """사격 플래그가 켜진 연속 구간의 (시작, 끝) 인덱스."""
    on = flag > 0.5
    segs, start = [], None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        if not v and start is not None:
            segs.append((start, i)); start = None
    if start is not None:
        segs.append((start, len(on)))
    return segs


def trajectory_figure(traj: np.ndarray, events=(), path: str | None = None,
                      title: str = "", blue_name: str = "Blue",
                      red_name: str = "Red", ec: EngagementConfig | None = None):
    ec = ec or EngagementConfig()
    tr = np.asarray(traj, dtype=np.float64)
    t = tr[:, _C["t"]]
    bx, by, bh = tr[:, _C["bx"]], tr[:, _C["by"]], tr[:, _C["bh"]]
    rx, ry, rh = tr[:, _C["rx"]], tr[:, _C["ry"]], tr[:, _C["rh"]]
    rng_ = tr[:, _C["range"]]
    ata = _ata_series(tr)

    fig = plt.figure(figsize=(11, 4.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], hspace=0.35, wspace=0.28)
    ax_xy = fig.add_subplot(gs[:, 0])
    ax_h = fig.add_subplot(gs[0, 1])
    ax_g = fig.add_subplot(gs[1, 1], sharex=ax_h)

    # (a) 평면 궤적 — 축은 (동, 북) 으로 두어 지도와 같은 방향
    for x, y, fl, col, nm in ((bx, by, tr[:, _C["fire_blue"]], BLUE, blue_name),
                              (rx, ry, tr[:, _C["fire_red"]], RED, red_name)):
        ax_xy.plot(y / 1000, x / 1000, color=col, lw=1.1, alpha=0.85, label=nm)
        for s, e in _fire_segments(t, fl):
            ax_xy.plot(y[s:e] / 1000, x[s:e] / 1000, color=col, lw=3.2, alpha=0.9)
        ax_xy.plot(y[0] / 1000, x[0] / 1000, "o", color=col, ms=6, mec="k", mew=0.6)
        ax_xy.plot(y[-1] / 1000, x[-1] / 1000, "x", color=col, ms=8, mew=2)
    # 시간 눈금: 20초마다 작은 점
    for tk in np.arange(20, t[-1], 20):
        i = int(np.searchsorted(t, tk))
        if i < len(t):
            ax_xy.plot(by[i] / 1000, bx[i] / 1000, ".", color=BLUE, ms=4)
            ax_xy.plot(ry[i] / 1000, rx[i] / 1000, ".", color=RED, ms=4)
    ax_xy.set_xlabel(T("East [km]", "동 [km]")); ax_xy.set_ylabel(T("North [km]", "북 [km]"))
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.grid(alpha=0.3); ax_xy.legend(fontsize=8, loc="best")
    ax_xy.set_title(T("(a) Plan view  (thick = in WEZ, firing)", "(a) 평면 궤적  (굵은 선 = WEZ 안, 사격 중)"), fontsize=9)

    # (b) 고도
    ax_h.plot(t, bh / 1000, color=BLUE, lw=1.2)
    ax_h.plot(t, rh / 1000, color=RED, lw=1.2)
    ax_h.axhline(ec.hard_deck / 1000, color="k", ls=":", lw=1)
    ax_h.text(t[-1], ec.hard_deck / 1000, T(" hard deck", " 하드덱"), fontsize=7, va="bottom", ha="right")
    ax_h.set_ylabel(T("Altitude [km]", "고도 [km]")); ax_h.grid(alpha=0.3)
    ax_h.set_title(T("(b) Altitude", "(b) 고도"), fontsize=9)
    plt.setp(ax_h.get_xticklabels(), visible=False)

    # (c) 거리 + ATA (이중축)
    ax_g.axhspan(ec.gun_range_min / 1000, ec.gun_range_max / 1000,
                 color="0.85", alpha=0.6, lw=0, label=T("gun range", "기총 사거리"))
    ax_g.plot(t, rng_ / 1000, color="k", lw=1.2, label=T("range", "거리"))
    ax_g.set_ylabel(T("Range [km]", "거리 [km]")); ax_g.set_xlabel(T("Time [s]", "시간 [s]"))
    ax_g.set_ylim(0, max(2.0, np.nanmax(rng_) / 1000 * 1.05))
    ax2 = ax_g.twinx()
    ax2.plot(t, ata, color=BLUE, lw=1.0, alpha=0.8, label=T("ATA (blue)", "ATA (청)"))
    cone = ec.cone_deg_start + (ec.cone_deg_end - ec.cone_deg_start) * np.clip(t / ec.episode_time, 0, 1)
    ax2.plot(t, cone, color=BLUE, ls="--", lw=0.8, alpha=0.6, label=T("hit cone", "명중 원추"))
    ax2.set_ylim(0, 180); ax2.set_ylabel(T("ATA [deg]", "ATA [도]"), color=BLUE)
    ax2.tick_params(axis="y", colors=BLUE)
    h1, l1 = ax_g.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax_g.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right", ncol=2)
    ax_g.grid(alpha=0.3)
    ax_g.set_title(T("(c) Range and blue ATA", "(c) 거리와 청군 ATA"), fontsize=9)

    # 이벤트 표시 (종료·WEZ 진입) — 고도 패널 위에 세로선
    for e in events:
        te, txt = float(e[0]), str(e[-1])
        if "종료" in txt or "WEZ" in txt:
            ax_h.axvline(te, color="0.5", lw=0.6, ls="-", alpha=0.6)

    if title:
        fig.suptitle(title, fontsize=10)
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return path
    return fig
