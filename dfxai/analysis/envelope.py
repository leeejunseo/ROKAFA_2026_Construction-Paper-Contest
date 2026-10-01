"""
기체 모델의 선회 성능 (논문 <그림 3-2>, 3.1.2절 물리적 타당성).

시뮬레이터의 점질량 모델(dynamics.py)이 속도별로 낼 수 있는 선회율을 계산합니다.

  순간 선회율  : 가용 하중배수 n_avail(양력 한계와 구조 한계 중 작은 값)로 수평 선회할 때
                 omega = g * sqrt(n^2 - 1) / V
  지속 선회율  : 최대추력에서 추력 = 항력이 되는 하중배수 n_s 로 수평 선회할 때.
                 T = q S (CD0 + k CL^2), CL = n W / (q S)  ->  n_s = (qS/W) sqrt((T/(qS) - CD0) / k)
                 (n_s 는 n_avail 을 넘을 수 없음)
  코너 속도    : 양력 한계가 구조 한계와 같아지는 속도, Vc = sqrt(2 n_max W / (rho S CLmax))

수식은 dynamics.py 와 같은 함수(air_density, max_thrust, max_load_factor)를 그대로 써서
"논문에 보고한 값 = 실험에 쓴 모델의 값"이 되게 했습니다.

실행:  python -m dfxai.analysis.envelope --outdir results/envelope
출력:  envelope.csv (속도별 값), envelope_summary.json, fig_turn_envelope.png, fig_turn_envelope_ko.png
"""
from __future__ import annotations
import argparse, json, math, os
import numpy as np
import pandas as pd

from ..config import AircraftConfig, G0
from ..dynamics import air_density, max_thrust, max_load_factor
from .labels import T, set_language, language

FT = 0.3048
KT = 0.514444            # 1 kt [m/s]
ALTS = {"SL": 0.0, "15,000 ft": 15_000 * FT}   # 공개 F-16 에너지 기동 선도에서 흔히 쓰는 고도
COLORS = {"SL": "#2a78d6", "15,000 ft": "#eb6834"}  # 범주형 1·2번 (dataviz 기본 팔레트)


def corner_speed(h: float, ac: AircraftConfig) -> float:
    w = ac.mass * G0
    return math.sqrt(2.0 * ac.n_max_struct * w / (air_density(h) * ac.wing_area * ac.cl_max))


def sustained_load_factor(v: float, h: float, ac: AircraftConfig) -> float:
    q = 0.5 * air_density(h) * v * v
    qs = q * ac.wing_area
    t = max_thrust(h, ac)
    if t / qs <= ac.cd0:
        return 0.0
    n_s = (qs / (ac.mass * G0)) * math.sqrt((t / qs - ac.cd0) / ac.k_induced)
    return min(n_s, max_load_factor(v, h, ac))


def turn_rate_deg(n: float, v: float) -> float:
    return math.degrees(G0 * math.sqrt(max(n * n - 1.0, 0.0)) / v) if n > 1.0 else 0.0


def compute(ac: AircraftConfig | None = None, v_grid=None) -> tuple[pd.DataFrame, dict]:
    ac = ac or AircraftConfig()
    v_grid = np.arange(80.0, 400.1, 1.0) if v_grid is None else np.asarray(v_grid)
    rows, summ = [], {}
    for name, h in ALTS.items():
        for v in v_grid:
            n_i = max_load_factor(v, h, ac)
            n_s = sustained_load_factor(v, h, ac)
            rows.append(dict(alt=name, h_m=h, v_ms=v, v_kt=v / KT,
                             n_inst=n_i, n_sust=n_s,
                             itr_dps=turn_rate_deg(n_i, v), str_dps=turn_rate_deg(n_s, v)))
        d = pd.DataFrame([r for r in rows if r["alt"] == name])
        vc = corner_speed(h, ac)
        i_s = int(d["str_dps"].idxmax())
        summ[name] = dict(
            h_m=h,
            corner_speed_ms=vc, corner_speed_ktas=vc / KT,
            itr_at_corner_dps=turn_rate_deg(ac.n_max_struct, vc),
            max_str_dps=float(d.loc[i_s, "str_dps"]),
            max_str_speed_ms=float(d.loc[i_s, "v_ms"]),
            max_str_speed_ktas=float(d.loc[i_s, "v_kt"]),
            max_str_load_factor=float(d.loc[i_s, "n_sust"]),
        )
    return pd.DataFrame(rows), summ


def plot(df: pd.DataFrame, summ: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name in ALTS:
        d = df[df["alt"] == name]
        c = COLORS[name]
        lab = T("sea level", "해면") if name == "SL" else T("15,000 ft", "15,000 ft (4,572 m)")
        ax.plot(d["v_ms"], d["itr_dps"], color=c, lw=2, label=f"{lab} · {T('instantaneous', '순간')}")
        ax.plot(d["v_ms"], d["str_dps"], color=c, lw=2, ls="--", label=f"{lab} · {T('sustained', '지속')}")
        s = summ[name]
        ax.plot([s["corner_speed_ms"]], [s["itr_at_corner_dps"]], "o", ms=8, color=c,
                markeredgecolor="#fcfcfb", markeredgewidth=2, zorder=5)
        ax.annotate(T(f"corner {s['corner_speed_ms']:.0f} m/s\n{s['itr_at_corner_dps']:.1f} deg/s",
                      f"코너 {s['corner_speed_ms']:.0f} m/s\n{s['itr_at_corner_dps']:.1f}°/s"),
                    (s["corner_speed_ms"], s["itr_at_corner_dps"]),
                    xytext=(-18, -4) if name == "SL" else (10, 8),
                    ha="right" if name == "SL" else "left", va="top" if name == "SL" else "bottom",
                    textcoords="offset points", fontsize=8, color="#0b0b0b")
    ax.set_xlabel(T("true airspeed [m/s]", "진대기속도 [m/s]"))
    ax.set_ylabel(T("turn rate [deg/s]", "선회율 [°/s]"))
    ax.set_xlim(80, 400); ax.set_ylim(0, 32)
    ax.grid(alpha=0.25, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="기체 모델 선회 성능 (그림 3-2)")
    ap.add_argument("--outdir", default="results/envelope")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    df, summ = compute()
    df.to_csv(os.path.join(a.outdir, "envelope.csv"), index=False)
    with open(os.path.join(a.outdir, "envelope_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=2, ensure_ascii=False)
    for lang, sfx in (("en", ""), ("ko", "_ko")):
        set_language(lang)
        plot(df, summ, os.path.join(a.outdir, f"fig_turn_envelope{sfx}.png"))
    print(json.dumps(summ, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
