"""
R9 비행역학 민감도 정리 -> paper/results_r9.md

같은 26개 정책(BT 3종, 300세대 순수 학습 20개, BTO 3개)을 비행역학 파라미터 하나씩만
바꾼 환경(results/r9_*)에서 다시 평가한 결과를 본실험(results/main, 같은 조건·같은 초기조건)과
비교합니다. 상대(BT-v2, BT-v3)도 같은 바뀐 기체로 비행합니다.

보는 것
  1) 정책 점수 순위가 유지되는가        ρ(본실험 점수, 변형 점수)
  2) D* 순위가 유지되는가               ρ(본실험 D*, 변형 D*)
  3) 성능–D* 무상관이 유지되는가        변형 안의 ρ(점수, D*)와 부트스트랩 90% 구간
  4) 단순화가 규칙 기반에 유리했는가    계열별 점수 변화, 학습 정책과 BT-v2 의 점수 차

실행:  python -m dfxai.analysis.dyn_sensitivity --out paper/results_r9.md
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
import pandas as pd

from .figures import load_merged, read_json
from .paper import md_table, _spearman

LABELS = {
    "cl080": "최대 양력계수 ×0.8", "cl120": "최대 양력계수 ×1.2",
    "thr080": "최대추력 ×0.8", "thr120": "최대추력 ×1.2",
    "lag080": "지령 지연 시상수 ×0.8", "lag120": "지령 지연 시상수 ×1.2",
    "roll180": "롤 속도 상한 180°/s", "roll090": "롤 속도 상한 90°/s",
}
ORDER = list(LABELS)
N_BOOT = 5000


def _fam(cond: str) -> str:
    return "BTO" if cond.startswith("BTO") else ("BT" if cond.startswith("BT-") else "RL")


def _boot_ci(x, y, q=0.90, seed=0):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    n, out = len(x), []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        if np.std(x[i]) < 1e-12 or np.std(y[i]) < 1e-12:
            continue
        out.append(pd.Series(x[i]).corr(pd.Series(y[i]), method="spearman"))
    return np.quantile(out, [(1 - q) / 2, (1 + q) / 2])


def summarize(d: pd.DataFrame) -> dict:
    d = d.copy(); d["fam"] = d["cond"].map(_fam)
    rl, bto = d[d.fam == "RL"], d[d.fam == "BTO"]
    g = lambda c, col: float(d.loc[d.cond == c, col].iloc[0])
    r, p, n = _spearman(d["score"], d["d_star"])
    lo, hi = _boot_ci(d["score"], d["d_star"])
    return {"n": n, "ρ(점수, D*)": r, "p": p, "90% 구간": f"[{lo:.2f}, {hi:.2f}]",
            "D*=17(절단)": int((d["d_star"] >= 17).sum()),
            "학습 평균 점수": rl["score"].mean(), "BT-v2 점수": g("BT-v2", "score"),
            "BT-v3 점수": g("BT-v3", "score"), "BTO 평균 점수": bto["score"].mean(),
            "학습−BT-v2": rl["score"].mean() - g("BT-v2", "score"),
            "BT-v2 초과 학습 정책": f"{int((rl['score'] > g('BT-v2', 'score')).sum())}/{len(rl)}",
            "학습 평균 D*": rl["d_star"].mean(), "BT-v2 D*": g("BT-v2", "d_star"),
            "BT-v3 D*": g("BT-v3", "d_star")}


def build(main_dir: str, out_path: str, roll_diag: str = "results/r9_roll_diag.csv") -> str:
    main = load_merged(main_dir)
    dirs = {os.path.basename(p)[3:]: p for p in glob.glob("results/r9_*") if os.path.isdir(p)}
    names = [k for k in ORDER if k in dirs and os.path.exists(os.path.join(dirs[k], "merged.csv"))]
    if not names:
        raise SystemExit("results/r9_*/merged.csv 가 없습니다")
    conds = load_merged(dirs[names[0]])["cond"].tolist()
    base = main[main["cond"].isin(conds)][["cond", "score", "d_star"]].reset_index(drop=True)

    rows_a, rows_b = [], [dict(설정="기준(본실험)", **summarize(base))]
    for k in names:
        s = load_merged(dirs[k])[["cond", "score", "d_star"]]
        man = read_json(os.path.join(dirs[k], "manifest.json"))
        both = base.merge(s, on="cond", suffixes=("_b", "_s"))
        r_s, p_s, n = _spearman(both["score_b"], both["score_s"])
        r_d, p_d, _ = _spearman(both["d_star_b"], both["d_star_s"])
        both["fam"] = both["cond"].map(_fam)
        dsc = (both["score_s"] - both["score_b"]).groupby(both["fam"]).mean()
        rows_a.append({"설정": LABELS[k], "env_opts": json.dumps(man.get("env_opts", {})), "n": n,
                       "ρ(점수 기준, 점수 변형)": r_s, "ρ(D* 기준, D* 변형)": r_d,
                       "Δ점수 BT 3종": dsc.get("BT", np.nan), "Δ점수 학습 20개": dsc.get("RL", np.nan),
                       "Δ점수 BTO 3개": dsc.get("BTO", np.nan),
                       "D* 같은 정책 수": f"{int((both['d_star_b'] == both['d_star_s']).sum())}/{n}"})
        rows_b.append(dict(설정=LABELS[k], **summarize(s)))

    # 8개 변형의 변형 내 ρ(점수, D*) 검정에 Holm 보정 (기준 행은 제외)
    pv = np.array([r["p"] for r in rows_b[1:]])
    order, adj, run = np.argsort(pv), np.empty(len(pv)), 0.0
    for i, j in enumerate(order):
        run = max(run, (len(pv) - i) * pv[j]); adj[j] = min(1.0, run)
    rows_b[0]["p (Holm, 8개 변형)"] = np.nan
    for r, q in zip(rows_b[1:], adj):
        r["p (Holm, 8개 변형)"] = q
    rows_b = [{k: r[k] for k in list(r)[:4] + ["p (Holm, 8개 변형)"] + list(r)[4:-1]} for r in rows_b]

    out = ["# R9. 비행역학 민감도 (자동 생성)", "",
           "R7(교전 규칙 민감도)과 같은 26개 정책(BT 3종, 300세대 순수 학습 20개, BTO 3개)을 같은 초기조건"
           "(시드 10000–10099 × 상대 BT-v2·BT-v3 × 진영 교대 = 조건당 400교전)으로, 비행역학 파라미터를 "
           "**하나씩만** 바꾼 환경에서 다시 평가했다. 상대도 같은 바뀐 기체로 비행한다. 정책은 다시 학습하지 "
           "않았고, 정책이 안에 품은 기체 가정(BT 의 코너속도 계산 등)도 그대로다. 기준은 본실험(`results/main`)의 "
           "같은 26개 조건이며, 바꾸지 않은 설정으로 다시 돌리면 본실험과 교전 단위까지 똑같이 재현됨을 확인했다.",
           "", "실행: `bash results/r9_pipeline.sh` → `python -m dfxai.analysis.dyn_sensitivity`", "",
           "## 1. 순위 유지와 계열별 점수 변화", "",
           "Δ점수는 (변형 점수 − 기준 점수)의 계열 평균이다.", "",
           md_table(pd.DataFrame(rows_a)), "",
           "## 2. 설정별 성능–설명가능성 관계와 계열 비교", "",
           "ρ(점수, D*)는 26개 정책의 Spearman 순위상관, 90% 구간은 정책 단위 부트스트랩"
           f"({N_BOOT}회)이다.", "",
           md_table(pd.DataFrame(rows_b), {"p": "{:.4f}", "p (Holm, 8개 변형)": "{:.3f}"}), ""]
    if os.path.exists(roll_diag):
        rd = pd.read_csv(roll_diag)
        rd["계열"] = rd["cond"].map(_fam)
        g = rd.groupby("계열")[["p_median", "p99", "p_max", "frac_over_90", "frac_over_180",
                                "frac_long_way"]].mean().reset_index()
        g.columns = ["계열", "롤 속도 중앙값 [°/s]", "롤 속도 99% [°/s]", "최대 [°/s]",
                     ">90°/s 비율", ">180°/s 비율", "먼 길 롤 비율"]
        out += ["## 3. 본실험 환경의 롤 속도 (상한 없음)", "",
                "본실험 설정에서 평가 대상 기체의 적분 스텝(0.05 s)별 롤 속도. 정책마다 상대 2종 × 초기조건 20개"
                "(청군 측만). '먼 길 롤'은 뱅크 지령과 현재 뱅크의 차가 180°를 넘어, 최단 경로가 아니라 수평을 "
                "거쳐 반대로 도는 적분 스텝의 비율이다(`python -m dfxai.analysis.roll_diag`).", "",
                md_table(g, {">90°/s 비율": "{:.3f}", ">180°/s 비율": "{:.3f}", "먼 길 롤 비율": "{:.3f}"}), ""]
    text = "\n".join(out)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def main():
    ap = argparse.ArgumentParser(description="R9 비행역학 민감도 정리")
    ap.add_argument("--main-dir", default="results/main")
    ap.add_argument("--out", default="paper/results_r9.md")
    a = ap.parse_args()
    print(build(a.main_dir, a.out))


if __name__ == "__main__":
    main()
