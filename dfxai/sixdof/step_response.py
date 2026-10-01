"""
내부 추종 제어기 사전 점검: 계단 응답 (설계안 3절, 부록용).

같은 지령을 3자유도(1차 지연)와 6자유도(JSBSim F-16 + 추종 제어기)에 똑같이 주고
응답을 비교한다. 교전 전에 이것을 보는 이유: 제어기가 지령을 따르지 못하면 6자유도에서
성능이 달라졌을 때 정책 탓인지 제어기 탓인지 가를 수 없다.

  뱅크 계단 0 -> 60°, 0 -> 135°  (하중 지령 1 G 유지)   : 상승시간, 오버슈트, 최대 롤 속도
  하중 계단 1 -> 5 G, 1 -> 8 G   (뱅크 지령 0° 유지)     : 상승시간, 오버슈트, 3초 안 최대값

실행:  python -m dfxai.sixdof.step_response --outdir results/sixdof
출력:  step_response.csv, step_metrics.csv, fig_step_response(_ko).png
"""
from __future__ import annotations
import argparse, math, os
import numpy as np
import pandas as pd

from ..config import AircraftConfig
from ..dynamics import AircraftState, step as dyn_step
from .aircraft import JSBAircraft

DT = 0.05
T_END = 4.0
SPEEDS = (180.0, 220.0, 260.0)
ALT = 4000.0
TESTS = [("bank", 60.0), ("bank", 135.0), ("load", 5.0), ("load", 8.0)]


def _cmd(kind, target):
    return ([math.radians(target), 1.0, 1.0] if kind == "bank" else [0.0, target, 1.0])


def run_3dof(kind, target, v):
    ac = AircraftConfig(); s = AircraftState(0, 0, ALT, v, 0.0); s.thr = 1.0
    out = []
    for k in range(int(T_END / DT)):
        dyn_step(s, _cmd(kind, target), DT, ac)
        out.append((round((k + 1) * DT, 3), math.degrees(s.mu), s.n))
    return out


def run_6dof(kind, target, v, ac6: JSBAircraft):
    s = AircraftState(0, 0, ALT, v, 0.0)
    ac6.reset_from(s); ac6.sync(s)
    out = []
    for k in range(int(T_END / DT)):
        ac6.step(_cmd(kind, target), DT); ac6.sync(s)
        out.append((round((k + 1) * DT, 3), math.degrees(s.mu), s.n))
    return out


def metrics(tr, kind, target):
    t = np.array([r[0] for r in tr]); y = np.array([r[1] if kind == "bank" else r[2] for r in tr])
    y0 = 0.0 if kind == "bank" else 1.0
    span = target - y0
    frac = (y - y0) / span
    def first(th):
        i = np.where(frac >= th)[0]
        return float(t[i[0]]) if i.size else np.nan
    t10, t90 = first(0.1), first(0.9)
    m = dict(rise_10_90=t90 - t10 if np.isfinite(t10) and np.isfinite(t90) else np.nan,
             t90=t90, overshoot_pct=float(max(0.0, (frac.max() - 1.0) * 100)),
             peak=float(y.max()), final=float(y[-1]))
    if kind == "bank":
        m["max_roll_rate"] = float(np.max(np.abs(np.diff(y) / DT)))
    else:
        m["min_value"] = float(y.min())
    return m


def run_all(ac6: JSBAircraft | None = None):
    ac6 = ac6 or JSBAircraft()
    rows, mets = [], []
    for v in SPEEDS:
        for kind, target in TESTS:
            for model, tr in (("3DOF", run_3dof(kind, target, v)), ("6DOF", run_6dof(kind, target, v, ac6))):
                rows += [dict(model=model, v=v, test=f"{kind}{target:g}", t=a, mu_deg=b, n=c) for a, b, c in tr]
                mets.append(dict(model=model, v=v, test=f"{kind}{target:g}", **metrics(tr, kind, target)))
    return pd.DataFrame(rows), pd.DataFrame(mets)


def plot(df: pd.DataFrame, path: str, v: float = 220.0) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..analysis.labels import T
    c3, c6 = "#2a78d6", "#eb6834"
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    for ax, kind, col, ylab in ((axes[0], "bank", "mu_deg", T("bank angle [deg]", "뱅크각 [°]")),
                                (axes[1], "load", "n", T("load factor [G]", "하중배수 [G]"))):
        for test in sorted({t for t in df.test if t.startswith(kind)}):
            target = float(test[len(kind):])
            for model, c, ls in (("3DOF", c3, "--"), ("6DOF", c6, "-")):
                d = df[(df.model == model) & (df.v == v) & (df.test == test)]
                ax.plot(d.t, d[col], color=c, ls=ls, lw=2,
                        label=(T("3DOF (1st-order lag)", "3자유도 (1차 지연)") if model == "3DOF"
                               else T("6DOF (JSBSim F-16)", "6자유도 (JSBSim F-16)")) if test == min(
                            t for t in df.test if t.startswith(kind)) else None)
            ax.axhline(target, color="#52514e", lw=0.8, ls=":")
        ax.set_xlabel(T("time [s]", "시간 [s]")); ax.set_ylabel(ylab)
        ax.grid(alpha=0.25, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="추종 제어기 계단 응답")
    ap.add_argument("--outdir", default="results/sixdof")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    df, met = run_all()
    df.to_csv(os.path.join(a.outdir, "step_response.csv"), index=False)
    met.to_csv(os.path.join(a.outdir, "step_metrics.csv"), index=False)
    from ..analysis.labels import set_language
    for lang, sfx in (("en", ""), ("ko", "_ko")):
        set_language(lang)
        plot(df, os.path.join(a.outdir, f"fig_step_response{sfx}.png"))
    pd.set_option("display.width", 200)
    print(met.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
