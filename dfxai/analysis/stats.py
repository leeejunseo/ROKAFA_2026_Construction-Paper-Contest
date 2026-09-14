"""
통계 분석.

계획서에 적으신 대로 승률을 주 지표로 삼고, 차이가 우연인지 검정하며,
여러 조건을 동시에 비교할 때 생기는 착시를 보정합니다.

주의할 점 두 가지
-----------------
1) 무승부 처리. 이 환경은 무승부가 적지 않게 나옵니다. 무승부를 0.5로
   환산한 '점수'와, 무승부를 제외한 '순수 승률'을 **둘 다** 보고하세요.
   한쪽만 쓰면 결론이 처리방식에 의존한다는 지적을 받습니다.

2) 짝지은 설계. 같은 시드로 진영을 바꿔 두 번 싸우므로 두 교전은
   독립이 아닙니다. 시드 단위로 두 결과를 평균한 뒤 그 시드 평균들을
   표본으로 삼아야 검정의 가정이 지켜집니다(clustered data).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def wilson_interval(k: float, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Wilson 점수 신뢰구간. 표본이 작거나 비율이 0/1에 가까울 때
    정규근사보다 훨씬 안정적입니다."""
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(0.5 + conf / 2)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((c - h) / d), float((c + h) / d)


def seed_level_scores(df: pd.DataFrame) -> pd.DataFrame:
    """시드 단위로 진영 교대 두 판을 평균 -> 독립 표본 생성."""
    g = (df.groupby(["cond", "opponent", "seed"], as_index=False)
           .agg(score=("score", "mean"), win=("win", "mean"),
                loss=("loss", "mean"), draw=("draw", "mean"),
                wez=("wez_time", "mean"), dmg=("damage_dealt", "mean"),
                surv=("surv_time", "mean"),
                viol_deck=("viol_deck", "mean"),
                viol_over_g=("viol_over_g", "mean"),
                viol_sep=("viol_sep", "mean")))
    return g


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """조건 x 상대별 요약표. 논문 결과표의 원본이 됩니다."""
    s = seed_level_scores(df)
    rows = []
    for (cond, opp), g in s.groupby(["cond", "opponent"]):
        n = len(g)
        score = g["score"].mean()
        lo, hi = wilson_interval(score * n, n)
        decisive = g["win"].sum() + g["loss"].sum()
        pure_wr = g["win"].sum() / decisive if decisive > 0 else np.nan
        rows.append(dict(
            cond=cond, opponent=opp, n_seeds=n,
            score=score, score_lo=lo, score_hi=hi,
            pure_winrate=pure_wr, draw_rate=g["draw"].mean(),
            wez_time=g["wez"].mean(), damage=g["dmg"].mean(),
            surv_time=g["surv"].mean(),
            viol_deck=g["viol_deck"].mean(),
            viol_over_g=g["viol_over_g"].mean(),
            viol_total=(g["viol_deck"] + g["viol_over_g"] + g["viol_sep"]).mean(),
        ))
    return pd.DataFrame(rows).sort_values(["opponent", "score"], ascending=[True, False])


def holm(pvals: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Holm-Bonferroni 보정. Bonferroni보다 검정력이 높으면서
    family-wise error rate를 동일하게 통제합니다."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (m - i) * p[idx])
        adj[idx] = min(1.0, running)
    return adj, adj <= alpha


def pairwise_tests(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """조건 간 쌍별 비교.

    같은 시드에서 두 조건이 같은 초기조건을 경험하므로
    **대응표본(paired) Wilcoxon 부호순위 검정**을 씁니다.
    독립표본 검정보다 검정력이 훨씬 높습니다.
    """
    s = seed_level_scores(df)
    out = []
    for opp, go in s.groupby("opponent"):
        piv = go.pivot_table(index="seed", columns="cond", values="score")
        conds = list(piv.columns)
        for i in range(len(conds)):
            for j in range(i + 1, len(conds)):
                a, b = piv[conds[i]].dropna(), piv[conds[j]].dropna()
                idx = a.index.intersection(b.index)
                a, b = a.loc[idx], b.loc[idx]
                if len(idx) < 8 or np.allclose(a.values, b.values):
                    p = 1.0
                else:
                    p = float(stats.wilcoxon(a.values, b.values,
                                             zero_method="zsplit").pvalue)
                out.append(dict(opponent=opp, cond_a=conds[i], cond_b=conds[j],
                                mean_a=a.mean(), mean_b=b.mean(),
                                diff=a.mean() - b.mean(), n=len(idx), p_raw=p))
    res = pd.DataFrame(out)
    if len(res):
        adj, sig = holm(res["p_raw"].values, alpha)
        res["p_holm"] = adj
        res["significant"] = sig
    return res


def binomial_vs_chance(df: pd.DataFrame) -> pd.DataFrame:
    """각 조건이 '동전던지기'와 다른지 검정 (무승부 제외 순수 승률 기준)."""
    s = seed_level_scores(df)
    rows = []
    for (cond, opp), g in s.groupby(["cond", "opponent"]):
        w = int(round(g["win"].sum() * 2))      # 진영 2판 복원
        l = int(round(g["loss"].sum() * 2))
        if w + l == 0:
            continue
        p = float(stats.binomtest(w, w + l, 0.5).pvalue)
        rows.append(dict(cond=cond, opponent=opp, wins=w, losses=l,
                         pure_winrate=w / (w + l), p_raw=p))
    res = pd.DataFrame(rows)
    if len(res):
        adj, sig = holm(res["p_raw"].values)
        res["p_holm"] = adj
        res["significant"] = sig
    return res
