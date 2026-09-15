"""
공통 상태 집합에서 잰 설명가능성 (common-state D*) 과 상태 방문 범위.

문제의식
--------
본실험의 D* 는 각 정책이 **스스로 방문한** 상태에서 대리트리를 적합한다. 그래서
'한 번 쏘고 이탈' 정책처럼 단조로운 궤적을 그리는 정책은 상태 분포가 좁아 D* 가
작게 나오고, '끝까지 추격' 정책은 온갖 기하를 겪어 D* 가 크게 나올 수 있다.
즉 D* 차이가 규칙의 복잡성인지 상황의 다양성인지 분리되지 않는다.

해결
----
(1) 모든 조건의 로그를 합쳐 균일 추출한 **공통 관측 집합**을 만들고, 각 정책에
    같은 관측을 질의해 얻은 행동으로 대리트리를 적합한다 → D*_common.
    세 계열 모두 관측 벡터만 받으면 행동을 내므로 방문하지 않은 상태에서도
    "그 상황이면 무엇을 할 것인가"를 물을 수 있다.
(2) 각 정책의 자기 로그에서 **상태 방문 범위**(거리·총각·애스펙트각 3차원
    히스토그램의 엔트로피)를 계산해 공변량으로 보고한다.

출력: results/<run>/xai_common.csv
  cond, d_star_common, d_star_090_common, f4_common, coverage_entropy, coverage_std
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np
import pandas as pd
from multiprocessing import Pool

from ..config import FEATURE_NAMES
from ..experiments.run_eval import build_condition
from ..analysis.figures import read_json
from .surrogate import surrogate_fidelity

_I_R, _I_ATA, _I_AA = 0, 3, 4
_I_ALPHA = FEATURE_NAMES.index("alpha")


def build_common_pool(outdir: str, n: int = 40_000, seed: int = 0) -> np.ndarray:
    """모든 조건의 궤적을 합쳐 균일 추출한 관측 집합."""
    rng = np.random.default_rng(seed)
    chunks = []
    for p in sorted(glob.glob(os.path.join(outdir, "traj_*.npz"))):
        ob = np.load(p)["obs"]
        k = min(len(ob), max(2000, n // 20))
        chunks.append(ob[rng.choice(len(ob), k, replace=False)])
    pool = np.concatenate(chunks, axis=0)
    if len(pool) > n:
        pool = pool[rng.choice(len(pool), n, replace=False)]
    return pool


def coverage_stats(obs: np.ndarray, bins: int = 10) -> dict:
    """상태 방문 범위. 거리·총각·애스펙트각 3차원 히스토그램의 정규화 엔트로피
    (0 = 한 칸에 몰림, 1 = 완전 균일) 와 관측 표준편차의 평균."""
    x = np.stack([np.clip(obs[:, _I_R], 0, 2) / 2.0, obs[:, _I_ATA], obs[:, _I_AA]], axis=1)
    h, _ = np.histogramdd(x, bins=bins, range=[(0, 1)] * 3)
    p = h.ravel() / max(1.0, h.sum())
    p = p[p > 0]
    ent = float(-(p * np.log(p)).sum() / np.log(bins ** 3))
    return dict(coverage_entropy=ent,
                coverage_std=float(obs[:, :19].std(axis=0).mean()))


def _job(args):
    spec, pool, alpha, target = args
    pol = build_condition(spec)
    obs = pool.copy()
    obs[:, _I_ALPHA] = alpha
    act = np.asarray([pol.act(o) for o in obs], dtype=np.float64)
    r = surrogate_fidelity(obs, act, target=target)
    return dict(cond=spec["name"], d_star_common=r.min_depth_at_target,
                d_star_090_common=r.d_star_multi["0.9"],
                d_star_099_common=r.d_star_multi["0.99"],
                f4_common=r.fidelity_at_fixed_depth)


def compute(outdir: str, n_pool: int = 40_000, workers: int = 1,
            target: float = 0.95, seed: int = 0) -> pd.DataFrame:
    man = read_json(os.path.join(outdir, "manifest.json"))
    pool = build_common_pool(outdir, n_pool, seed)
    np.save(os.path.join(outdir, "common_pool.npy"), pool)
    jobs = []
    for spec in man["conditions"]:
        alpha = float(spec.get("alpha", 1.0 if spec["kind"] in ("rl", "shield") else 0.0))
        jobs.append((spec, pool, alpha, target))
    if workers > 1:
        with Pool(workers) as p:
            rows = p.map(_job, jobs)
    else:
        rows = [_job(j) for j in jobs]
    df = pd.DataFrame(rows)

    cov = []
    for spec in man["conditions"]:
        p = os.path.join(outdir, f"traj_{spec['name']}.npz")
        if os.path.exists(p):
            c = coverage_stats(np.load(p)["obs"]); c["cond"] = spec["name"]; cov.append(c)
    df = df.merge(pd.DataFrame(cov), on="cond", how="left")
    df.to_csv(os.path.join(outdir, "xai_common.csv"), index=False)
    return df


def main():
    ap = argparse.ArgumentParser(description="공통 상태 집합 D* 와 상태 방문 범위")
    ap.add_argument("--outdir", default="results/main")
    ap.add_argument("--n-pool", type=int, default=40_000)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    df = compute(a.outdir, a.n_pool, a.workers)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
