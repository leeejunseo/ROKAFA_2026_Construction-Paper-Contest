"""
진화전략(OpenAI-ES) 학습기 — numpy만 있으면 돌아갑니다.

왜 이걸 같이 넣었나:
  - torch / SB3 설치 없이 **당장 오늘** 학습 정책을 뽑을 수 있습니다.
  - 하이퍼파라미터가 거의 없고, 병렬화가 자명하며, 발산하지 않습니다.
    PPO 튜닝에 며칠 날리는 위험이 없어 '플랜 B'로 확실합니다.
  - 설명가능성 축에서 문제가 되는 것은 "블랙박스 신경망 정책"이지
    "그것이 PPO로 학습되었는가"가 아닙니다. 논문에서는 학습 알고리즘을
    명시하고 모델을 '학습 기반 모델'로 부르면 논지가 그대로 유지됩니다.

PPO 경로는 rl/train_ppo.py 를 쓰세요. 두 경로 모두 같은 MLPPolicy 포맷으로
체크포인트를 저장하므로 평가·분석 파이프라인은 공유됩니다.

가장 중요한 설계: **여러 세대에서 체크포인트를 저장**합니다.
이 체크포인트들이 논문의 '학습 예산' 축이 되고,
RL이 BT를 못 이기더라도 성능-설명가능성 곡선을 그릴 수 있게 해 줍니다.
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
from multiprocessing import Pool

from ..config import OBS_DIM
from ..env import DogfightEnv, run_episode
from ..agents.bt import BTPolicy
from ..agents.hybrid import MLPPolicy, HybridPolicy
from .shaping import potential, episode_fitness


# ----------------------------------------------------------------- 적합도
def _rollout_fitness(args):
    """후보 파라미터 하나에 대한 평균 적합도."""
    flat, hidden, seeds, alphas, opp_versions, bt_version = args
    rl = MLPPolicy(hidden=hidden, params=flat)
    bt = BTPolicy(version=bt_version)
    env = DogfightEnv()
    total = 0.0
    for seed, alpha, ov in zip(seeds, alphas, opp_versions):
        blue = HybridPolicy(bt, rl, alpha=alpha)
        red = BTPolicy(version=ov)
        ob, orr = env.reset(seed=int(seed), alpha=float(alpha))
        pot_sum, n = 0.0, 0
        for _ in range(env.max_steps + 1):
            a_b, a_r = blue.act(ob), red.act(orr)
            ob, orr, done = env.step(a_b, a_r)
            pot_sum += potential(ob); n += 1
            if done:
                break
        total += episode_fitness(env.result(), pot_sum / max(n, 1))
    return total / len(seeds)


def _rank_transform(x: np.ndarray) -> np.ndarray:
    """순위 기반 정규화. 적합도 스케일 변화에 둔감해져 학습이 안정됩니다."""
    ranks = np.empty_like(x)
    ranks[np.argsort(x)] = np.arange(len(x))
    y = ranks / (len(x) - 1) - 0.5
    return y / (y.std() + 1e-8)


# ----------------------------------------------------------------- 학습 루프
def train(generations=200, pop=40, sigma=0.08, lr=0.03, episodes=4,
          hidden=(32, 32), seed=0, workers=1, bt_version=2,
          opponents=(1, 2), outdir="results/es", checkpoint_every=20,
          alpha_curriculum=True, tag="seed0"):
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)
    proto = MLPPolicy(hidden=hidden, seed=seed)
    theta = proto.flat.copy() * 0.1          # 작은 초기값에서 출발
    n_par = theta.size
    history = []
    pool = Pool(workers) if workers > 1 else None
    t0 = time.time()

    for gen in range(1, generations + 1):
        # --- alpha 커리큘럼: 초반엔 BT 비중을 높여 탐색을 유도 ---
        if alpha_curriculum:
            a_lo = min(0.9, 0.15 + 0.85 * gen / max(1, generations * 0.7))
        else:
            a_lo = 0.0
        alphas = rng.uniform(a_lo, 1.0, episodes)
        alphas[-1] = 1.0                      # 순수 RL 구간을 항상 1개 포함
        # 세대 내 모든 후보가 **같은 시드**를 쓰도록 고정 (분산 감소, 매우 중요)
        seeds = rng.integers(0, 10 ** 6, episodes)
        opps = rng.choice(opponents, episodes)

        eps = rng.normal(0.0, 1.0, (pop // 2, n_par))
        eps = np.concatenate([eps, -eps], axis=0)      # 미러 샘플링
        cands = theta[None, :] + sigma * eps

        jobs = [(cands[i], hidden, seeds, alphas, opps, bt_version)
                for i in range(pop)]
        fits = (np.array(pool.map(_rollout_fitness, jobs)) if pool
                else np.array([_rollout_fitness(j) for j in jobs]))

        grad = (_rank_transform(fits)[:, None] * eps).mean(axis=0) / sigma
        theta = theta + lr * grad

        history.append(dict(gen=gen, fit_mean=float(fits.mean()),
                            fit_max=float(fits.max()), alpha_lo=float(a_lo),
                            elapsed=time.time() - t0))
        if gen % max(1, generations // 20) == 0 or gen == 1:
            print(f"[gen {gen:4d}] fit mean={fits.mean():8.3f} "
                  f"max={fits.max():8.3f} alpha>={a_lo:.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

        if gen % checkpoint_every == 0 or gen == generations:
            p = MLPPolicy(hidden=hidden, params=theta)
            p.save(os.path.join(outdir, f"ckpt_{tag}_gen{gen:05d}.npz"))

    if pool:
        pool.close(); pool.join()
    with open(os.path.join(outdir, f"history_{tag}.json"), "w") as f:
        json.dump(history, f, indent=2)
    return theta, history


def main():
    ap = argparse.ArgumentParser(description="ES 학습기 (numpy 전용)")
    ap.add_argument("--generations", type=int, default=200)
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--sigma", type=float, default=0.08)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--bt-version", type=int, default=2)
    ap.add_argument("--outdir", type=str, default="results/es")
    ap.add_argument("--checkpoint-every", type=int, default=20)
    ap.add_argument("--tag", type=str, default=None)
    args = ap.parse_args()
    tag = args.tag or f"seed{args.seed}"
    train(generations=args.generations, pop=args.pop, sigma=args.sigma,
          lr=args.lr, episodes=args.episodes, seed=args.seed,
          workers=args.workers, bt_version=args.bt_version,
          outdir=args.outdir, checkpoint_every=args.checkpoint_every, tag=tag)


if __name__ == "__main__":
    main()
