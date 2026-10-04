"""
6자유도 평가 전용 교차검증: 실행과 판정 (사전 기준: paper/6DOF_교차검증_사전기준_초안.md, 2026-10-01 확정).

정책 집합 (안 A, 31개) — 이름·체크포인트·설정은 3자유도 본실험과 똑같다.
  BT-v1/v2/v3, 300세대 순수 학습 20개(RL-s0..19-b300-a1.00), BTO-s0..2-b300,
  선형 혼합 HYB-s11-b300-a0.50, Shield SHD-s11-b300, 잔차형 RES-s0-b200, 게이팅형 GATE-s1-b200,
  PPO-s0-b1500000
3자유도 기준값: results/main (29개), results/main_hyb (RES-s0-b200, GATE-s1-b200; 같은 시드·상대)

판정 (대안 1)
  C1 점수 순위 유지      : Spearman ρ(3DOF 점수, 6DOF 점수) ≥ 0.7
  C2 D* 순위 유지        : Spearman ρ(3DOF D*, 6DOF D*) ≥ 0.7
  C3 트레이드오프 출현   : 6DOF ρ(점수, D*) 의 90% 부트스트랩 구간 하한 > 0  그리고
                           6DOF ρ − 3DOF ρ(같은 31개) > 0.2.   둘 다 성립하면 "출현", 아니면 "3자유도와 같은 관계"
                           (정책 단위 재표집 5000회, 난수 시드 0)
  C4 규칙 준수 구조 유지 : BT-v3·SHD-s11 의 하드덱 위반 0 (판정). 과G·최소이격 위반은 참고로 보고

실행:  python -m dfxai.sixdof.crossval run --workers 4      -> results/sixdof_main/
       python -m dfxai.analysis.report --outdir results/sixdof_main   (D* 계산)
       python -m dfxai.sixdof.crossval judge                -> paper/results_6dof.md
"""
from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd

OUTDIR = "results/sixdof_main"
N_BOOT = 5000


def conditions() -> list[dict]:
    c = [dict(name=f"BT-v{v}", kind="bt", bt_version=v) for v in (1, 2, 3)]
    c += [dict(name=f"RL-s{s}-b300-a1.00", kind="rl", alpha=1.0, ckpt=f"results/es/ckpt_seed{s}_gen00300.npz",
               train_seed=s, bt_version=2, budget=300) for s in range(20)]
    c += [dict(name=f"BTO-s{s}-b300", kind="bto", ckpt=f"results/es_bto/ckpt_seed{s}_gen00300.npz",
               train_seed=s, budget=300, alpha=0.0) for s in range(3)]
    c += [dict(name="HYB-s11-b300-a0.50", kind="hybrid", alpha=0.5, ckpt="results/es/ckpt_seed11_gen00300.npz",
               train_seed=11, bt_version=2, budget=300),
          dict(name="SHD-s11-b300", kind="shield", ckpt="results/es/ckpt_seed11_gen00300.npz",
               train_seed=11, budget=300, alpha=1.0, bt_version=3),
          dict(name="RES-s0-b200", kind="residual", ckpt="results/es_res/ckpt_seed0_gen00200.npz",
               train_seed=0, budget=200, alpha=1.0, bt_version=2),
          dict(name="GATE-s1-b200", kind="gating", ckpt="results/es_gate/ckpt_seed1_gen00200.npz",
               train_seed=1, budget=200, alpha=1.0, bt_version=2),
          dict(name="PPO-s0-b1500000", kind="rl", ckpt="results/ppo/ckpt_seed0_step001500000.npz",
               train_seed=0, budget=1500000, alpha=1.0)]
    return c


def run(workers: int) -> None:
    from ..experiments.run_eval import run as run_eval
    run_eval(conditions(), opponents=(2, 3), n_seeds=100, workers=workers, outdir=OUTDIR,
             record_episodes=30, env_opts={"sixdof": True})


def _fam(c: str) -> str:
    for p, f in (("BTO", "BTO"), ("BT-", "BT"), ("RL-", "RL"), ("HYB", "선형 혼합"), ("SHD", "Shield"),
                 ("RES", "잔차형"), ("GATE", "게이팅형"), ("PPO", "PPO")):
        if c.startswith(p):
            return f
    return "?"


def _rho(x, y):
    from scipy.stats import spearmanr
    r, p = spearmanr(x, y)
    return float(r), float(p)


def _boot(x, y, q=0.90, seed=0):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    out = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(x), len(x))
        if np.std(x[i]) < 1e-12 or np.std(y[i]) < 1e-12:
            continue
        out.append(pd.Series(x[i]).corr(pd.Series(y[i]), method="spearman"))
    return np.quantile(out, [(1 - q) / 2, (1 + q) / 2])


def baseline_3dof() -> pd.DataFrame:
    names = [c["name"] for c in conditions()]
    m = pd.concat([pd.read_csv("results/main/merged.csv"),
                   pd.read_csv("results/main_hyb/merged.csv")]).drop_duplicates("cond")
    m = m[m.cond.isin(names)][["cond", "score", "d_star", "fidelity_at_4"]]
    e = pd.concat([pd.read_csv("results/main/episodes.csv"),
                   pd.read_csv("results/main_hyb/episodes.csv")])
    e = e[e.cond.isin(names)].drop_duplicates(["cond", "opponent", "seed", "side"])
    v = e.groupby("cond")[["viol_deck", "viol_over_g", "viol_sep"]].mean().reset_index()
    out = m.merge(v, on="cond")
    assert len(out) == len(names), f"3DOF 기준값 누락: {set(names) - set(out.cond)}"
    return out


def six_dof() -> pd.DataFrame:
    m = pd.read_csv(os.path.join(OUTDIR, "merged.csv"))[["cond", "score", "d_star", "fidelity_at_4"]]
    e = pd.read_csv(os.path.join(OUTDIR, "episodes.csv"))
    v = e.groupby("cond")[["viol_deck", "viol_over_g", "viol_sep"]].mean().reset_index()
    t = e.groupby("cond")["trim_ok"].mean().rename("trim_ok_frac").reset_index()
    return m.merge(v, on="cond").merge(t, on="cond")


def judge(out_path: str = "paper/results_6dof.md") -> str:
    from ..analysis.paper import md_table
    b, s = baseline_3dof(), six_dof()
    d = b.merge(s, on="cond", suffixes=("_3", "_6"))
    d["계열"] = d["cond"].map(_fam)
    n = len(d)
    c1, p1 = _rho(d.score_3, d.score_6)
    c2, p2 = _rho(d.d_star_3, d.d_star_6)
    r3, pr3 = _rho(d.score_3, d.d_star_3)
    r6, pr6 = _rho(d.score_6, d.d_star_6)
    lo3, hi3 = _boot(d.score_3, d.d_star_3)
    lo6, hi6 = _boot(d.score_6, d.d_star_6)
    c3_appear = bool(lo6 > 0 and (r6 - r3) > 0.2)
    rule = d[d.cond.isin(["BT-v3", "SHD-s11-b300"])]
    c4 = bool((rule.viol_deck_6 == 0).all())

    verdict = pd.DataFrame([
        {"기준": "C1 점수 순위 유지", "지표": "ρ(3DOF 점수, 6DOF 점수)", "값": f"{c1:.3f} (p={p1:.4f}, n={n})",
         "기준값": "≥ 0.7", "판정": "유지" if c1 >= 0.7 else "유지 안 됨"},
        {"기준": "C2 D* 순위 유지", "지표": "ρ(3DOF D*, 6DOF D*)", "값": f"{c2:.3f} (p={p2:.4f})",
         "기준값": "≥ 0.7", "판정": "유지" if c2 >= 0.7 else "유지 안 됨"},
        {"기준": "C3 트레이드오프 출현", "지표": "6DOF ρ(점수, D*) 구간 하한, 6DOF ρ − 3DOF ρ",
         "값": f"하한 {lo6:.3f}, 차이 {r6 - r3:+.3f}", "기준값": "하한 > 0 그리고 차이 > 0.2",
         "판정": "출현" if c3_appear else "3자유도와 같은 관계"},
        {"기준": "C4 규칙 준수 구조", "지표": "BT-v3·Shield 하드덱 위반율",
         "값": ", ".join(f"{r.cond} {r.viol_deck_6:.4f}" for r in rule.itertuples()), "기준값": "0",
         "판정": "유지" if c4 else "유지 안 됨"},
    ])
    corr = pd.DataFrame([
        {"환경": "3자유도", "ρ(점수, D*)": r3, "p": pr3, "90% 구간": f"[{lo3:.2f}, {hi3:.2f}]"},
        {"환경": "6자유도", "ρ(점수, D*)": r6, "p": pr6, "90% 구간": f"[{lo6:.2f}, {hi6:.2f}]"}])
    fam = d.groupby("계열").agg(n=("cond", "size"), 점수_3DOF=("score_3", "mean"), 점수_6DOF=("score_6", "mean"),
                              D_3DOF=("d_star_3", "mean"), D_6DOF=("d_star_6", "mean"),
                              하드덱_6DOF=("viol_deck_6", "mean"), 과G_6DOF=("viol_over_g_6", "mean")).reset_index()
    rl = d[d["계열"] == "RL"]
    bt2 = d[d.cond == "BT-v2"].iloc[0]
    per = d[["cond", "계열", "score_3", "score_6", "d_star_3", "d_star_6", "viol_deck_3", "viol_deck_6",
             "viol_over_g_3", "viol_over_g_6", "trim_ok_frac"]].sort_values(["계열", "cond"])

    text = "\n".join([
        "# 6자유도 평가 전용 교차검증 결과 (자동 생성)", "",
        "사전 기준: `paper/6DOF_교차검증_사전기준_초안.md`(2026-10-01 확정, 안 A·대안 1). "
        "정책을 다시 학습하지 않고, 3자유도에서 학습한 31개 정책을 JSBSim F-16(기체 FBW + 내부 추종 제어기)으로 평가했다. "
        "교전 규칙·관측·지령 변환·초기조건(시드 10000–10099 × 상대 BT-v2·BT-v3 × 진영 교대 = 정책당 400교전)은 3자유도와 같다.",
        "", f"실행: `python -m dfxai.sixdof.crossval run` → `python -m dfxai.analysis.report --outdir {OUTDIR}` → "
        "`python -m dfxai.sixdof.crossval judge`", "",
        "## 1. 사전 기준 판정", "", md_table(verdict), "",
        "## 2. 성능–설명가능성 상관 (같은 31개 정책)", "", md_table(corr, {"p": "{:.4f}"}), "",
        f"학습 정책 20개 평균 점수: 3자유도 {rl.score_3.mean():.3f} → 6자유도 {rl.score_6.mean():.3f}. "
        f"BT-v2: {bt2.score_3:.3f} → {bt2.score_6:.3f}. BT-v2 를 넘은 학습 정책: 3자유도 "
        f"{int((rl.score_3 > bt2.score_3).sum())}/20 → 6자유도 {int((rl.score_6 > bt2.score_6).sum())}/20.", "",
        "## 3. 계열별", "", md_table(fam), "",
        "## 4. 정책별", "", md_table(per), "",
        f"트림 성공률(전체 교전): {s.trim_ok_frac.mean():.4f}", ""])
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def main():
    ap = argparse.ArgumentParser(description="6자유도 교차검증")
    ap.add_argument("cmd", choices=("run", "judge", "baseline"))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.workers)
    elif a.cmd == "baseline":
        print(baseline_3dof().to_string())
    else:
        print(judge())


if __name__ == "__main__":
    main()
