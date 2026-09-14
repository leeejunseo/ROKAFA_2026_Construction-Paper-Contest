"""
그림 1: 실험 프레임워크 개요도.

    python -m dfxai.analysis.overview --out results/main/paper_figs/fig_overview.png --lang ko

환경(시뮬레이터) → 세 정책 계열(같은 관측·같은 지령) → 진영 교대 평가 →
세 평가축 의 흐름을 한 장에 그립니다. 논문 방법 절 첫 그림으로 쓰도록
설계했고, 수치(관측 20차원, 지령 3차원, 시드 100 등)는 config 와 실험 설계에서
그대로 가져왔습니다.
"""
from __future__ import annotations
import argparse, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from .labels import T, set_language

C_ENV, C_BT, C_RL, C_HYB, C_EVAL, C_AX = "#e8eef7", "#dbe9f6", "#f8dcd6", "#e2f0d9", "#f3f3f3", "#fff4d6"


def _box(ax, x, y, w, h, title, lines, color, title_size=10, size=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                                fc=color, ec="#555", lw=1.0))
    ax.text(x + w / 2, y + h - 0.045, title, ha="center", va="top", fontsize=title_size, weight="bold")
    ax.text(x + 0.03, y + h - 0.11, "\n".join(lines), ha="left", va="top", fontsize=size, linespacing=1.45)


def _arrow(ax, x0, y0, x1, y1, text=None, size=8):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14,
                                 lw=1.2, color="#333", connectionstyle="arc3,rad=0"))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.03, text, ha="center", va="bottom", fontsize=size, color="#333")


def draw(path: str, n_conds: int = 67, n_fights: int = 26_800, n_train_seeds: int = 5) -> str:
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # 1) 환경
    _box(ax, 0.02, 0.30, 0.20, 0.48,
         T("Dogfight simulator", "교전 시뮬레이터"),
         [T("3-DoF point-mass F-16", "3자유도 점질량 F-16급"),
          T("1 v 1, guns only, 120 s", "1대1 기총, 120 s"),
          T("WEZ 120–1,200 m, cone 3°→10°", "WEZ 120~1,200 m, 원추 3°→10°"),
          T("Hard deck 1,000 m · 8 G · 150 m", "하드덱 1,000 m · 8 G · 이격 150 m"),
          "",
          T("Observation: 20-D geometry", "관측: 교전기하 20차원"),
          T("(range, ATA, AA, closure, ΔE…)", "(거리·ATA·AA·접근률·에너지차…)"),
          T("Command: bank · G · throttle", "지령: 뱅크 · G · 스로틀 (3차원)")],
         C_ENV)

    # 2) 정책 세 계열 (세로로)
    _box(ax, 0.30, 0.66, 0.24, 0.20,
         T("Rule-based: behavior trees", "규칙 기반: 행동트리 3종"),
         [T("BT-v1 horizontal pursuit (8 nodes)", "BT-v1 수평 추격 (노드 8)"),
          T("BT-v2 3-D max-rate (5 nodes)", "BT-v2 3차원 최대선회 (노드 5)"),
          T("BT-v3 + safety rules (11 nodes)", "BT-v3 + 안전규칙 (노드 11)")],
         C_BT)
    _box(ax, 0.30, 0.42, 0.24, 0.20,
         T("Learned: MLP 20-32-32-3", "학습 기반: MLP 20-32-32-3"),
         [T("Init: behavior-cloned BT-v2", "초기화: BT-v2 행동 복제"),
          T("ES fine-tuning, accept test", "ES 미세조정 + 수용 검사"),
          T("Budget 0 / 100 / 200 / 300 gen", "예산 0 / 100 / 200 / 300 세대")],
         C_RL)
    _box(ax, 0.30, 0.18, 0.24, 0.20,
         T("Hybrid: mix dial α", "혼합: 혼합계수 α"),
         [r"$a=(1-\alpha)\,a_{BT}+\alpha\,a_{RL}$",
          T("α ∈ {0.25, 0.5, 0.75}", "α ∈ {0.25, 0.5, 0.75}"),
          T("circular mean on bank", "뱅크는 원형 평균")],
         C_HYB)
    ax.text(0.42, 0.90, T("Same observation · same command space", "같은 관측 · 같은 지령 공간"),
            ha="center", fontsize=9, style="italic", color="#333")

    # 3) 평가
    _box(ax, 0.62, 0.30, 0.17, 0.48,
         T("Paired evaluation", "진영 교대 짝지은 평가"),
         [T("Opponents: BT-v2 (trained on)", "상대: BT-v2 (학습에 사용)"),
          T("            BT-v3 (held out)", "      BT-v3 (보류 상대)"),
          T("100 seeds × 2 sides", "100시드 × 진영 2"),
          T("(same seed, swap sides)", "(같은 시드로 청/홍 교대)"),
          "",
          T(f"{n_conds} conditions · {n_fights:,} fights", f"{n_conds}조건 · {n_fights:,}교전"),
          T(f"{n_train_seeds} training seeds", f"학습 시드 {n_train_seeds}개"),
          T("Wilson CI · Wilcoxon · Holm", "Wilson CI · Wilcoxon · Holm")],
         C_EVAL)

    # 4) 세 평가축
    _box(ax, 0.86, 0.64, 0.12, 0.24, T("Performance", "전투 성능"),
         [T("score (W1/D½/L0)", "점수 (승1·무½·패0)"), T("pure win rate", "순수승률"),
          T("outcome mix", "종료 사유")], C_AX, title_size=9, size=7)
    _box(ax, 0.86, 0.38, 0.12, 0.24, T("Explainability", "설명가능성"),
         [T("surrogate tree", "대리 결정트리"), T("D* = min depth", "D* = 최소 깊이"),
          T("at fidelity 0.95", "(충실도 0.95)"), "F(4)"], C_AX, title_size=9, size=7)
    _box(ax, 0.86, 0.12, 0.12, 0.24, T("Rule compliance", "규칙 준수"),
         [T("hard deck", "하드덱 침범"), T("over-G", "과G"), T("separation", "최소이격"),
          T("per-step rate", "스텝당 위반율")], C_AX, title_size=9, size=7)

    # 화살표
    for y in (0.76, 0.52, 0.28):
        _arrow(ax, 0.22, 0.54 if y == 0.52 else y, 0.30, y)
        _arrow(ax, 0.54, y, 0.62, 0.54 if y == 0.52 else y)
        _arrow(ax, 0.79, 0.54 if y == 0.52 else y, 0.86, y)
    ax.text(0.26, 0.82, T("obs", "관측"), ha="center", fontsize=8, color="#333")
    ax.text(0.58, 0.82, T("action", "행동"), ha="center", fontsize=8, color="#333")
    ax.text(0.825, 0.82, T("logs", "로그"), ha="center", fontsize=8, color="#333")

    ax.text(0.5, 0.03, T(
        "Three policy families share one interface, one arena and one yardstick; "
        "training budget and mix dial α are the independent variables.",
        "세 계열이 같은 인터페이스·같은 환경·같은 잣대를 공유하며, 학습 예산과 혼합계수 α 가 독립변수다."),
        ha="center", fontsize=9, color="#333")

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description="실험 프레임워크 개요도")
    ap.add_argument("--out", default="results/main/paper_figs/fig_overview.png")
    ap.add_argument("--lang", default="en", choices=("en", "ko"))
    a = ap.parse_args()
    set_language(a.lang)
    print(draw(a.out))


if __name__ == "__main__":
    main()
