"""
합성 로그 생성기.

**이 스크립트가 일정 관리의 핵심입니다.**

학습이 끝나기를 기다렸다가 분석 코드를 짜기 시작하면, 학습이 하루
늦어질 때마다 논문도 하루 늦어집니다. 그래서 가짜 로그를 먼저 만들어
통계·대리모델·그림 파이프라인을 전부 완성해 두십시오. 그러면 실제
체크포인트가 나오는 순간 경로만 바꿔서 최종 결과를 뽑을 수 있습니다.

생성되는 것은 실제 실행 결과와 **동일한 스키마**의 episodes.csv 와
traj_*.npz 입니다. 다만 성능·설명가능성 관계를 인위적으로 심어 두었으므로
**합성 데이터의 수치를 논문에 절대 쓰지 마십시오.** 파이프라인 검증 전용입니다.

사용:
    python -m dfxai.experiments.make_synthetic --outdir results/dryrun
    python -m dfxai.analysis.report --outdir results/dryrun
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import pandas as pd

from ..config import OBS_DIM, ACTION_DIM, config_dump

# (조건명, 계열, 기대 승률, 행동 복잡도, 규칙 위반율)
SPEC = [
    ("BT-v1", "bt", 0.30, 1.0, 0.002),
    ("BT-v2", "bt", 0.58, 2.0, 0.050),
    ("BT-v3", "bt", 0.45, 3.0, 0.000),
    ("HYB-b1000000-a0.25", "hybrid", 0.47, 3.5, 0.004),
    ("HYB-b1000000-a0.50", "hybrid", 0.50, 5.0, 0.010),
    ("HYB-b1000000-a0.75", "hybrid", 0.53, 7.0, 0.020),
    ("RL-b1000000-a1.00", "rl", 0.52, 9.0, 0.035),
    ("HYB-b3000000-a0.50", "hybrid", 0.57, 6.0, 0.012),
    ("RL-b3000000-a1.00", "rl", 0.62, 11.0, 0.045),
    ("RL-b8000000-a1.00", "rl", 0.70, 14.0, 0.060),
]


def _fake_traj(complexity: float, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """복잡도에 비례해 더 깊은 트리를 요구하는 인공 행동 궤적.

    관측은 무작위, 행동은 관측의 비선형 함수입니다. complexity 가 클수록
    고주파 성분이 커져 얕은 트리로는 재현이 어려워집니다.
    """
    X = rng.normal(0, 0.6, (n, OBS_DIM))
    A = np.zeros((n, ACTION_DIM))
    k = 0.30 * complexity          # 트리 깊이 요구량이 1~12 범위에 들도록 조정
    A[:, 0] = np.tanh(2.0 * X[:, 1]) + 0.25 * np.sin(k * X[:, 3]) \
        + 0.15 * np.sin(k * 1.7 * X[:, 0] * X[:, 5])
    A[:, 1] = np.tanh(1.5 * X[:, 7]) + 0.20 * np.cos(k * X[:, 11])
    A[:, 2] = np.clip(0.9 + 0.05 * np.sin(k * X[:, 12]), -1, 1)
    return X, np.clip(A, -1, 1)


def generate(outdir: str, n_seeds: int = 100, opponents=(2, 3),
             n_traj: int = 12000, seed: int = 0) -> pd.DataFrame:
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for cond, fam, wr, cx, viol in SPEC:
        for ov in opponents:
            p = float(np.clip(wr - (0.04 if ov == max(opponents) else 0.0), 0.02, 0.98))
            for i in range(n_seeds):
                for side in (0, 1):
                    u = rng.random()
                    if u < p * 0.8:
                        sc, w, d, l, oc = 1.0, 1, 0, 0, "gun_kill"
                    elif u < p * 0.8 + 0.22:
                        sc, w, d, l, oc = 0.5, 0, 1, 0, "timeout"
                    else:
                        sc, w, d, l, oc = 0.0, 0, 0, 1, "gun_kill"
                    rows.append(dict(
                        cond=cond, kind=fam, alpha=0.0, bt_version=-1, ckpt="",
                        budget=-1, opponent=f"BT-v{ov}", seed=10_000 + i, side=side,
                        score=sc, win=w, draw=d, loss=l, outcome=oc,
                        duration=float(rng.uniform(30, 120)),
                        damage_dealt=float(rng.uniform(0, 100)),
                        damage_taken=float(rng.uniform(0, 100)),
                        wez_time=float(rng.uniform(0, 8)),
                        surv_time=float(rng.uniform(30, 120)),
                        viol_deck=float(max(0, rng.normal(viol, viol * 0.3 + 1e-4))),
                        viol_over_g=float(max(0, rng.normal(viol, viol * 0.3 + 1e-4))),
                        viol_sep=0.0, mean_es=float(rng.uniform(3000, 6000))))
        X, A = _fake_traj(cx, n_traj, rng)
        np.savez_compressed(os.path.join(outdir, f"traj_{cond}.npz"), obs=X, act=A)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "episodes.csv"), index=False)
    with open(os.path.join(outdir, "manifest.json"), "w") as f:
        json.dump({"SYNTHETIC": True,
                   "warning": "파이프라인 검증 전용. 논문에 쓰지 말 것.",
                   "config": config_dump()}, f, indent=2, ensure_ascii=False)
    print(f"합성 로그 생성 완료: {outdir}  (교전 {len(df)}행)")
    print("주의: 이 수치는 인위적으로 심은 것입니다. 논문에 쓰지 마세요.")
    return df


def main():
    ap = argparse.ArgumentParser(description="합성 로그 생성 (파이프라인 검증용)")
    ap.add_argument("--outdir", default="results/dryrun")
    ap.add_argument("--n-seeds", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    generate(a.outdir, a.n_seeds, seed=a.seed)


if __name__ == "__main__":
    main()
