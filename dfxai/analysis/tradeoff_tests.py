"""
트레이드오프 검증 두 가지 (논문 3.5절, 4.2절).

    python -m dfxai.analysis.tradeoff_tests --outdir results/main

1) 연관성 검정: 성능 정의 3종(학습 상대 BT-v2, 보류 상대 BT-v3, 상대 2종 평균) × 설명 비용 정의 3종
   (D*(0.95), D*(0.90), 1−F(4))의 Spearman ρ와 조건 부트스트랩 90% 구간. 모델 집합은 전체 조건과
   혼합 정책을 뺀 조건(BT·BTO·학습 기반·PPO). 전원 대전 26개 모델의 풀 점수도 같은 방식으로.
   → results/main/tradeoff_matrix.csv

2) 제약 비용 분석: D* ≤ k 인 모델 가운데 가장 강한 모델과 전체에서 가장 강한 모델의 점수 차.
   최고 모델을 고르는 데 쓴 초기조건과 점수 차를 재는 초기조건을 짝수/홀수로 나눠 사후 선택의
   부풀림을 없애고 양방향을 평균한다. 점수 차의 유의성은 초기조건 단위 Wilcoxon 부호순위 검정.
   → results/main/tradeoff_price_cv.csv
"""
from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon


def _ci(x, y, rng, n=2000):
    x, y = np.asarray(x), np.asarray(y); v = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        if np.ptp(x[i]) > 0 and np.ptp(y[i]) > 0:
            v.append(spearmanr(x[i], y[i]).statistic)
    return np.percentile(v, [5, 95])


def association(m: pd.DataFrame, pool: pd.DataFrame | None, rng) -> pd.DataFrame:
    subsets = {"전체": m, "혼합 제외": m[m.family.isin(["BT", "BTO", "RL", "PPO"])]}
    perfs = [("학습 상대 BT-v2", "vs_v2"), ("보류 상대 BT-v3", "vs_v3"), ("상대 2종 평균", "score")]
    costs = [("D*(0.95)", "d_star_095"), ("D*(0.90)", "d_star_090"), ("1-F(4)", "cost_f4")]
    rows = []
    for sn, d in subsets.items():
        for pn, pc in perfs:
            for cn, cc in costs:
                r, p = spearmanr(d[pc], d[cc]); lo, hi = _ci(d[pc], d[cc], rng)
                rows.append((sn, pn, cn, len(d), r, p, lo, hi))
    if pool is not None:
        d = m.merge(pool[["cond", "pool_score"]], on="cond")
        for cn, cc in costs:
            r, p = spearmanr(d["pool_score"], d[cc]); lo, hi = _ci(d["pool_score"], d[cc], rng)
            rows.append(("전원 대전", "풀 점수", cn, len(d), r, p, lo, hi))
    return pd.DataFrame(rows, columns=["집합", "성능", "설명비용", "n", "rho", "p", "ci90_lo", "ci90_hi"])


def price_cv(m: pd.DataFrame, episodes: pd.DataFrame, ks=(2, 3, 4, 5, 6, 8)) -> pd.DataFrame:
    mm = m.set_index("cond")
    per = episodes.groupby(["cond", "opponent", "seed"]).score.mean().unstack(["opponent", "seed"])
    seeds = sorted({c[1] for c in per.columns})
    half = {0: [s for s in seeds if s % 2 == 0], 1: [s for s in seeds if s % 2 == 1]}
    defs = {"학습 상대 BT-v2": ["BT-v2"], "보류 상대 BT-v3": ["BT-v3"], "상대 2종 평균": ["BT-v2", "BT-v3"]}
    rows = []
    for name, opps in defs.items():
        for cost in ("d_star_095", "d_star_090"):
            for k in ks:
                prices, diffs = [], []
                for sel, ev in ((0, 1), (1, 0)):
                    cols_s = [c for c in per.columns if c[0] in opps and c[1] in half[sel]]
                    cols_e = [c for c in per.columns if c[0] in opps and c[1] in half[ev]]
                    ps = per[cols_s].mean(axis=1)
                    allowed = mm.index[mm[cost] <= k]
                    b_k = ps.loc[ps.index.intersection(allowed)].idxmax()
                    b_all = ps.idxmax()
                    d = (per.loc[b_all, cols_e] - per.loc[b_k, cols_e]).values
                    prices.append(d.mean()); diffs.append(d)
                d = np.concatenate(diffs)
                p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
                rows.append((name, cost, k, float(np.mean(prices)), p))
    return pd.DataFrame(rows, columns=["성능", "설명비용", "k", "가격", "p"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results/main")
    ap.add_argument("--pool", default="results/pool/pool_scores.csv")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    m = pd.read_csv(os.path.join(a.outdir, "merged.csv"))
    m["vs_v2"] = 2 * m["score"] - m["score_heldout"]
    m["vs_v3"] = m["score_heldout"]
    m["cost_f4"] = 1 - m["fidelity_at_4"]
    pool = pd.read_csv(a.pool) if os.path.exists(a.pool) else None
    t1 = association(m, pool, rng)
    t1.to_csv(os.path.join(a.outdir, "tradeoff_matrix.csv"), index=False, encoding="utf-8")
    e = pd.read_csv(os.path.join(a.outdir, "episodes.csv"), usecols=["cond", "opponent", "seed", "score"])
    t2 = price_cv(m, e)
    t2.to_csv(os.path.join(a.outdir, "tradeoff_price_cv.csv"), index=False, encoding="utf-8")
    pd.set_option("display.width", 200)
    print(t1.round(3).to_string()); print(); print(t2.round(3).to_string())


if __name__ == "__main__":
    main()
