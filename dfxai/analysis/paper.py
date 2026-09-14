"""
논문 원고에 그대로 붙일 수 있는 Markdown 결과 문서를 생성합니다.

    python -m dfxai.analysis.paper --outdir results/main --esdir results/es \
                                   --out paper/results_auto.md

report.py 가 만든 summary/xai/pairwise/binomial.csv 와 manifest.json 을
읽어 다음을 씁니다.

  1. 실험 설정 표 (기체 제원·교전 규칙·관측 스케일)  -> 부록
  2. 모델 복잡도 표 (BT 노드 수·트리 구조, MLP 파라미터 수)  -> 방법
  3. 주 결과표: 조건별 성능·설명가능성·규칙준수  -> 결과 표 1
  4. 상관분석: 성능 vs D*, 성능 vs 위반율 (Spearman)  -> 결과 (논문 제목의 '상관관계')
  5. 학습예산 단조성 검정 (예산 vs 성능, 예산 vs D*)
  6. 쌍별 검정 요약 (Holm 보정)
  7. 대리모델 규칙 추출 예시 (깊이 3)  -> 정성적 예시
  8. 그림 목록

수치는 전부 CSV 에서 읽으므로 실험을 다시 돌리면 문서도 같이 갱신됩니다.
문장은 손으로 쓰되, 숫자는 이 파일에서 복사하십시오.
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
import pandas as pd
from scipy import stats

from ..config import config_dump, FEATURE_NAMES, ACTION_NAMES
from ..agents.bt import BTPolicy
from ..agents.hybrid import MLPPolicy
from ..xai.surrogate import rule_extract
from .figures import parse_cond, load_merged, read_json


# ------------------------------------------------------------ 도우미
def md_table(df: pd.DataFrame, fmt: dict | None = None, index: bool = False) -> str:
    """tabulate 의존성 없이 Markdown 표를 만듭니다."""
    fmt = fmt or {}
    d = df.reset_index() if index else df
    cols = list(d.columns)

    def cell(c, v):
        if isinstance(v, float):
            if np.isnan(v):
                return "–"
            f = fmt.get(c, "{:.3f}")
            return f.format(v)
        return str(v)

    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(cell(c, r[c]) for c in cols) + " |")
    return "\n".join(lines)


def _spearman(x, y) -> tuple[float, float, int]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 4:
        return np.nan, np.nan, int(ok.sum())
    r = stats.spearmanr(x[ok], y[ok])
    return float(r.statistic), float(r.pvalue), int(ok.sum())


def _kendall(x, y) -> tuple[float, float, int]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 4:
        return np.nan, np.nan, int(ok.sum())
    r = stats.kendalltau(x[ok], y[ok])
    return float(r.statistic), float(r.pvalue), int(ok.sum())


# ------------------------------------------------------------ 절 생성
def sec_config() -> str:
    c = config_dump()
    rows = []
    labels = {
        "aircraft": "기체(F-16급 점질량)", "engagement": "교전 규칙", "obs": "관측 스케일"}
    for grp in ("aircraft", "engagement", "obs"):
        for k, v in c[grp].items():
            rows.append(dict(구분=labels[grp], 항목=k, 값=str(v)))
    s = "### 부록 A. 실험 설정 (config.py 스냅샷)\n\n" + md_table(pd.DataFrame(rows))
    s += "\n\n관측 벡터 20차원: " + ", ".join(f"`{n}`" for n in FEATURE_NAMES)
    s += "\n\n행동 3차원: " + ", ".join(f"`{n}`" for n in ACTION_NAMES) + "\n"
    return s


def sec_complexity() -> str:
    rows = []
    trees = ""
    for v in (1, 2, 3):
        p = BTPolicy(version=v)
        rows.append(dict(모델=p.name, 종류="행동트리", 복잡도=f"노드 {p.complexity()['bt_nodes']}개",
                         설명={1: "합류 회피 + 수평 추격, 고정 4G",
                              2: "합류 회피 + 3차원 양력벡터 지향, 최대 G·추력",
                              3: "v2 + 하드덱 회복·8G 자율제한·리드추격"}[v]))
        trees += f"\n**{p.name}**\n\n```\n{p.describe()}```\n"
    mlp = MLPPolicy()
    rows.append(dict(모델="RL (MLP)", 종류="신경망", 복잡도=f"파라미터 {mlp.complexity()['nn_params']:,}개",
                     설명=f"20→{mlp.hidden[0]}→{mlp.hidden[1]}→3, tanh; ES 학습"))
    rows.append(dict(모델="Hybrid", 종류="혼합", 복잡도="BT + MLP",
                     설명="a = (1−α)·a_BT + α·a_RL, α∈{0.25,0.5,0.75}"))
    return ("### 표 M1. 모델 복잡도\n\n" + md_table(pd.DataFrame(rows)) +
            "\n\n행동트리 구조:\n" + trees)


def _load_all(outdir: str):
    summ = pd.read_csv(os.path.join(outdir, "summary.csv"))
    xai = pd.read_csv(os.path.join(outdir, "xai.csv"))
    pw = pd.read_csv(os.path.join(outdir, "pairwise.csv"))
    bino = pd.read_csv(os.path.join(outdir, "binomial.csv"))
    man = read_json(os.path.join(outdir, "manifest.json"))
    merged = load_merged(outdir)
    return summ, xai, merged, pw, bino, man


def sec_main_table(summ: pd.DataFrame, xai: pd.DataFrame, merged: pd.DataFrame,
                   man: dict, outdir: str) -> str:
    opps = sorted(summ["opponent"].unique())
    held = opps[-1]
    n_seeds = man.get("n_seeds", "?")
    out = (f"### 표 1. 조건별 전투성능·설명가능성·규칙준수\n\n"
           f"상대 {', '.join(opps)} 각각 {n_seeds}시드 × 진영 2 = {2 * n_seeds if isinstance(n_seeds, int) else '?'}교전. "
           f"점수 = 승 1 · 무 0.5 · 패 0 (시드 평균 기준 Wilson 95% CI). "
           f"순수승률 = 무승부 제외. D* = 충실도 0.95 도달 최소 트리 깊이 "
           f"(미달 시 17로 우측 절단). F(4) = 깊이 4 충실도. 위반율 = 적분 스텝당.\n\n")

    # 상대별 성능
    rows = []
    for cond in merged["cond"]:
        r = dict(조건=cond)
        for o in opps:
            s = summ[(summ["cond"] == cond) & (summ["opponent"] == o)]
            if len(s):
                s = s.iloc[0]
                r[f"점수 vs {o}"] = f"{s['score']:.3f} [{s['score_lo']:.2f}, {s['score_hi']:.2f}]"
                r[f"순수승률 vs {o}"] = f"{s['pure_winrate']:.3f}"
                r[f"무승부 vs {o}"] = f"{s['draw_rate']:.2f}"
        rows.append(r)
    out += "**(a) 전투 성능**\n\n" + md_table(pd.DataFrame(rows)) + "\n\n"

    # 종료 사유 분포 — 무승부의 정체(충돌인지 시간종료인지)를 밝혀야 합니다
    ep_path = os.path.join(outdir, "episodes.csv")
    if os.path.exists(ep_path):
        ep = pd.read_csv(ep_path)
        oc = (ep.groupby("cond")["outcome"].value_counts(normalize=True)
                .unstack(fill_value=0.0).reindex(merged["cond"]).reset_index())
        oc = oc.rename(columns={"gun_kill": "격추", "timeout": "시간종료",
                                "crash": "지면충돌", "collision": "공중충돌"})
        out += ("**(a′) 종료 사유 분포 (교전 비율, 진영·상대 합산)** — 무승부는 "
                "공중충돌(양측 패)과 HP 동률 시간종료로 나뉩니다.\n\n"
                + md_table(oc) + "\n\n")

    # 설명가능성 + 규칙준수 (상대 평균)
    rows = []
    for _, m in merged.iterrows():
        cond = m["cond"]
        s = summ[summ["cond"] == cond]
        x = xai[xai["cond"] == cond]
        r = dict(조건=cond, 계열=m["family"],
                 평균점수=m["score"], 보류상대점수=m.get("score_heldout", np.nan))
        if len(x):
            x = x.iloc[0]
            r.update({"D*(0.90)": x["d_star_090"], "D*(0.95)": x["d_star_095"],
                      "D*(0.99)": x["d_star_099"], "F(4)": x["fidelity_at_4"],
                      "표본수": int(x["n_samples"])})
        r.update(하드덱위반=s["viol_deck"].mean(), 과G위반=s["viol_over_g"].mean(),
                 위반합계=s["viol_total"].mean(), WEZ초=s["wez_time"].mean())
        rows.append(r)
    fmt = {"D*(0.90)": "{:.0f}", "D*(0.95)": "{:.0f}", "D*(0.99)": "{:.0f}",
           "하드덱위반": "{:.4f}", "과G위반": "{:.4f}", "위반합계": "{:.4f}", "WEZ초": "{:.2f}"}
    out += "**(b) 설명가능성·규칙준수 (상대 2종 평균)**\n\n" + md_table(pd.DataFrame(rows), fmt) + "\n"
    out += f"\n보류 상대 = {held} (학습에 사용하지 않은 상대).\n"
    return out


def sec_correlation(merged: pd.DataFrame) -> str:
    out = "### 표 2. 성능–설명가능성–규칙준수 상관 (Spearman ρ / Kendall τ)\n\n"
    out += ("조건을 표본 단위로 한 순위상관입니다. D* 는 우측 절단(17)을 포함하므로 "
            "순위 기반 상관이 적절하며, 절단값은 '가장 큼'으로 취급됩니다.\n\n")
    rows = []
    subsets = [("전체 조건", merged),
               ("학습 정책만 (RL+Hybrid)", merged[merged["family"].isin(["RL", "Hybrid"])]),
               ("BT 3종만", merged[merged["family"] == "BT"])]
    pairs = [("score", "d_star", "점수 vs D*(0.95)"),
             ("score", "fidelity_at_4", "점수 vs F(4)"),
             ("score", "viol_total", "점수 vs 위반율"),
             ("d_star", "viol_total", "D* vs 위반율")]
    for nm, d in subsets:
        for a, b, lab in pairs:
            if a not in d.columns or b not in d.columns:
                continue
            rho, p, n = _spearman(d[a], d[b])
            tau, pt, _ = _kendall(d[a], d[b])
            rows.append(dict(부분집합=nm, 비교=lab, n=n, Spearman_rho=rho, p_rho=p,
                             Kendall_tau=tau, p_tau=pt))
    out += md_table(pd.DataFrame(rows), {"p_rho": "{:.4f}", "p_tau": "{:.4f}"}) + "\n"
    out += ("\n해석 지침: ρ(점수, D*) > 0 이면 '성능이 높을수록 더 깊은 규칙이 필요'하다는 "
            "트레이드오프 방향입니다. BT 3종은 n=3 이라 검정력이 없으므로 방향만 보고하세요.\n")
    return out


def sec_budget_monotonic(merged: pd.DataFrame) -> str:
    d = merged[merged["family"].isin(["RL", "Hybrid"])].dropna(subset=["budget"])
    if d.empty or d["budget"].nunique() < 2:
        return "### 표 3. 학습예산 단조성\n\n(학습 체크포인트가 2개 미만이라 생략)\n"
    out = "### 표 3. 학습예산에 따른 단조 관계 (alpha 별 Spearman)\n\n"
    out += ("같은 alpha 안에서 학습 세대 수(예산)가 늘 때 성능과 D* 가 함께 오르는지 봅니다. "
            "학습 시드가 여럿이면 (예산, 시드) 조합이 표본입니다.\n\n")
    rows = []
    for al, g in d.groupby("alpha"):
        r1, p1, n = _spearman(g["budget"], g["score"])
        r2, p2, _ = _spearman(g["budget"], g["d_star"])
        r3, p3, _ = _spearman(g["budget"], g["viol_total"])
        rows.append({"alpha": al, "n": n, "ρ(예산, 점수)": r1, "p_점수": p1,
                     "ρ(예산, D*)": r2, "p_D*": p2, "ρ(예산, 위반율)": r3, "p_위반율": p3})
    r1, p1, n = _spearman(d["budget"], d["score"])
    r2, p2, _ = _spearman(d["budget"], d["d_star"])
    r3, p3, _ = _spearman(d["budget"], d["viol_total"])
    rows.append({"alpha": "전체", "n": n, "ρ(예산, 점수)": r1, "p_점수": p1,
                 "ρ(예산, D*)": r2, "p_D*": p2, "ρ(예산, 위반율)": r3, "p_위반율": p3})
    out += md_table(pd.DataFrame(rows), {"p_점수": "{:.4f}", "p_D*": "{:.4f}", "p_위반율": "{:.4f}"}) + "\n"

    # 예산×alpha 평균표 (학습 시드 평균)
    g = d.groupby(["budget", "alpha"]).agg(
        점수=("score", "mean"), 점수_sd=("score", "std"), Dstar=("d_star", "mean"),
        Dstar_sd=("d_star", "std"), F4=("fidelity_at_4", "mean"),
        위반율=("viol_total", "mean"), n_seed=("score", "size")).reset_index()
    out += "\n**예산 × alpha 평균 (학습 시드 평균 ± 표준편차)**\n\n"
    out += md_table(g, {"위반율": "{:.4f}", "budget": "{:.0f}"}) + "\n"
    return out


def sec_pairwise(pw: pd.DataFrame, bino: pd.DataFrame) -> str:
    out = "### 표 4. 통계 검정\n\n"
    if len(bino):
        out += "**(a) 동전던지기(0.5) 대비 순수승률 이항검정, Holm 보정**\n\n"
        b = bino.sort_values(["opponent", "pure_winrate"], ascending=[True, False])
        out += md_table(b, {"p_raw": "{:.4f}", "p_holm": "{:.4f}"}) + "\n\n"
    if len(pw):
        out += "**(b) 조건 간 대응표본 Wilcoxon (시드 단위 짝지음), Holm 보정 후 유의한 쌍**\n\n"
        sig = pw[pw["significant"] == True].sort_values(["opponent", "p_holm"])
        if len(sig):
            out += md_table(sig[["opponent", "cond_a", "cond_b", "mean_a", "mean_b", "diff", "n", "p_holm"]],
                            {"p_holm": "{:.4f}"}) + "\n"
        else:
            out += "(Holm 보정 후 유의한 쌍 없음)\n"
        out += f"\n총 {len(pw)}쌍 검정, 유의 {int(pw['significant'].sum())}쌍. 전체는 pairwise.csv 참조.\n"
    return out


def sec_rules(outdir: str, merged: pd.DataFrame, depth: int = 3) -> str:
    out = f"### 정성적 예시. 깊이 {depth} 대리 결정트리 규칙\n\n"
    out += ("각 정책의 (관측, 행동) 로그에 깊이 3 회귀트리를 적합한 결과입니다. "
            "value 는 [bank_cmd, load_cmd, throttle_cmd] (정규화 [-1,1]) 입니다. "
            "BT 는 몇 개 조건으로 행동이 거의 결정되는 반면 학습 정책은 같은 깊이에서 "
            "잔차가 크게 남는다는 점을 본문에서 대비시키십시오.\n\n")
    picks = list(merged[merged["family"] == "BT"]["cond"])
    learn = merged[merged["family"].isin(["RL", "Hybrid"])]
    if len(learn):
        maxb = learn["budget"].max()
        top = learn[learn["budget"] == maxb]
        for fam in ("Hybrid", "RL"):
            t = top[top["family"] == fam]
            if len(t):
                picks.append(t.sort_values("score", ascending=False).iloc[0]["cond"])
    for cond in picks:
        p = os.path.join(outdir, f"traj_{cond}.npz")
        if not os.path.exists(p):
            continue
        d = np.load(p)
        obs, act = d["obs"], d["act"]
        if obs.shape[0] > 30_000:
            idx = np.linspace(0, obs.shape[0] - 1, 30_000).astype(int)
            obs, act = obs[idx], act[idx]
        txt = rule_extract(obs, act, depth=depth)
        out += f"**{cond}** (표본 {obs.shape[0]:,})\n\n```\n{txt}```\n\n"
    return out


def sec_figures(outdir: str) -> str:
    figs = sorted(glob.glob(os.path.join(outdir, "fig_*.png")) +
                  glob.glob(os.path.join(outdir, "paper_figs", "*.png")))
    out = "### 그림 목록\n\n"
    desc = {
        "fig_pareto": "성능–설명가능성 평면(조건별) + 파레토 프론티어",
        "fig_compliance": "성능–규칙준수 평면",
        "fig_fidelity": "깊이별 대리모델 충실도 곡선",
        "fig_learning_curve": "ES 학습곡선",
        "fig_budget_sweep": "학습예산 × alpha 에 따른 성능/D*/위반율 (핵심 그림)",
        "fig_tradeoff_families": "성능–설명가능성 평면(예산·alpha 집계, 시드 오차막대)",
        "fig_traj_": "대표 교전 궤적 (같은 seed, 다른 정책)",
    }
    for f in figs:
        b = os.path.basename(f)
        d = next((v for k, v in desc.items() if b.startswith(k)), "")
        out += f"- `{os.path.relpath(f)}` — {d}\n"
    return out


def sec_training(esdir: str) -> str:
    files = sorted(glob.glob(os.path.join(esdir, "history_*.json")))
    if not files:
        return ""
    rows = []
    for f in files:
        h = pd.DataFrame(read_json(f))
        tag = os.path.basename(f)[len("history_"):-len(".json")]
        rows.append(dict(학습시드=tag, 세대=int(h["gen"].max()),
                         초기적합도=float(h["fit_mean"].iloc[0]),
                         최종적합도=float(h["fit_mean"].iloc[-5:].mean()),
                         최대적합도=float(h["fit_max"].max()),
                         소요시간_분=float(h["elapsed"].iloc[-1] / 60)))
    return ("### 표 M2. 학습 실행 요약 (ES)\n\n" + md_table(pd.DataFrame(rows), {"소요시간_분": "{:.1f}"}) +
            "\n\n초기·최종 적합도는 해당 세대에서 현재 정책 θ 를 같은 시드 8교전으로 평가한 값"
            "(최종은 마지막 5세대 평균)이며, 세대마다 시드가 바뀌므로 잡음이 큽니다. "
            "학습이 됐는지는 이 표가 아니라 표 3 의 예산별 점수로 판단하십시오. "
            "실행 설정: BT-v2 행동 복제 초기화, OpenAI-ES 변형(미러 샘플링·순위 정규화·"
            "갱신 수용 검사), pop 32, σ 0.05, lr 0.01, 후보당 8교전(α 격자 {0.25,0.5,0.75,1.0} × 2), "
            "상대 BT-v1/v2 교대. 명령은 README 를 보십시오.\n")


def build(outdir: str, esdir: str, out_path: str) -> str:
    summ, xai, merged, pw, bino, man = _load_all(outdir)
    synthetic = man.get("SYNTHETIC", False)
    parts = ["# 실험 결과 자동 생성 문서\n",
             f"원본: `{outdir}` · 조건 {len(merged)}개 · 상대 {man.get('opponents')} · "
             f"시드 {man.get('n_seeds')}\n"]
    if synthetic:
        parts.append("\n> **경고: 합성 데이터입니다. 수치를 논문에 쓰지 마십시오.**\n")
    parts += [sec_main_table(summ, xai, merged, man, outdir), sec_correlation(merged),
              sec_budget_monotonic(merged), sec_pairwise(pw, bino),
              sec_rules(outdir, merged), sec_figures(outdir),
              "\n---\n", sec_complexity(), sec_training(esdir), sec_config()]
    doc = "\n".join(parts)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="논문용 Markdown 결과 문서 생성")
    ap.add_argument("--outdir", default="results/main")
    ap.add_argument("--esdir", default="results/es")
    ap.add_argument("--out", default="paper/results_auto.md")
    a = ap.parse_args()
    print(build(a.outdir, a.esdir, a.out))


if __name__ == "__main__":
    main()
