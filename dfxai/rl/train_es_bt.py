"""
상수 최적화 행동트리(BTO) 학습기.

bt_param.ParamBTPolicy 의 상수 9개를 학습 정책과 **같은 적합도·같은 예산**
(300세대 × 개체군 32 × 후보당 8교전, BT-v1/v2 상대 교대, 갱신 수용 검사)으로
진화전략 최적화한다. 트리 구조와 노드 수는 고정이므로 결과물은 여전히 사람이
읽을 수 있는 규칙이다.

학습 정책과 다른 점은 탐색 파라미터뿐이다. 신경망은 1,827차원 가중치 공간이라
σ 0.05·lr 0.01 을 썼고, BTO 는 [-1, 1] 로 정규화한 9차원이라 σ 0.10·lr 0.05 를
쓴다(차원이 다르면 같은 보폭이 같은 의미가 아니다). 이 차이는 논문에 명시한다.
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
from multiprocessing import Pool

from ..env import DogfightEnv
from ..agents.bt import BTPolicy
from ..agents.bt_param import ParamBTPolicy, default_theta, N_PARAMS
from .shaping import potential, episode_fitness
from .train_es import _rank_transform, _dump_history


def _rollout(args):
    theta, seeds, opps, preset = args
    pol = ParamBTPolicy(theta=theta)
    env = DogfightEnv()
    total = 0.0
    for seed, ov in zip(seeds, opps):
        red = BTPolicy(version=int(ov))
        ob, orr = env.reset(seed=int(seed), alpha=0.0)
        pot_sum, n = 0.0, 0
        for _ in range(env.max_steps + 1):
            ob, orr, done = env.step(pol.act(ob), red.act(orr))
            pot_sum += potential(ob); n += 1
            if done:
                break
        total += episode_fitness(env.result(), pot_sum / max(n, 1), preset)
    return total / len(seeds)


def train(generations=300, pop=32, sigma=0.10, lr=0.05, episodes=8, seed=0,
          workers=1, opponents=(1, 2), outdir="results/es_bto",
          checkpoint_every=50, tag="seed0", fitness="default"):
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)
    theta = default_theta() + rng.normal(0.0, 0.05, N_PARAMS)   # BT-v3 근방에서 출발
    lr_eff, n_accept, history = lr, 0, []
    pool = Pool(workers) if workers > 1 else None
    t0 = time.time()
    for gen in range(1, generations + 1):
        seeds = rng.integers(0, 10 ** 6, episodes)
        opps = np.array([opponents[i % len(opponents)] for i in range(episodes)])
        rng.shuffle(opps)
        eps = rng.normal(0.0, 1.0, (pop // 2, N_PARAMS))
        eps = np.concatenate([eps, -eps], axis=0)
        cands = np.clip(theta[None, :] + sigma * eps, -1.0, 1.0)
        jobs = [(cands[i], seeds, opps, fitness) for i in range(pop)]
        fits = (np.array(pool.map(_rollout, jobs)) if pool
                else np.array([_rollout(j) for j in jobs]))
        grad = (_rank_transform(fits)[:, None] * eps).mean(axis=0) / sigma
        theta_new = np.clip(theta + lr_eff * grad, -1.0, 1.0)
        chk = [(theta, seeds, opps, fitness), (theta_new, seeds, opps, fitness)]
        f_old, f_new = (pool.map(_rollout, chk) if pool else [_rollout(j) for j in chk])
        accepted = f_new >= f_old
        if accepted:
            theta = theta_new; lr_eff = min(lr, lr_eff * 1.15)
        else:
            lr_eff = max(lr * 0.25, lr_eff * 0.7)
        n_accept += int(accepted)
        history.append(dict(gen=gen, fit_mean=float(fits.mean()), fit_max=float(fits.max()),
                            fit_theta=float(f_new if accepted else f_old),
                            accepted=bool(accepted), lr_eff=float(lr_eff),
                            elapsed=time.time() - t0, alpha_lo=0.0))
        if gen % max(1, generations // 20) == 0 or gen == 1:
            print(f"[gen {gen:4d}] fit mean={fits.mean():8.3f} max={fits.max():8.3f} "
                  f"theta={history[-1]['fit_theta']:8.3f} accept={n_accept}/{gen} "
                  f"lr={lr_eff:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if gen % checkpoint_every == 0 or gen == generations:
            ParamBTPolicy(theta=theta).save(os.path.join(outdir, f"ckpt_{tag}_gen{gen:05d}.npz"))
            _dump_history(outdir, tag, history)
    if pool:
        pool.close(); pool.join()
    p = ParamBTPolicy(theta=theta)
    with open(os.path.join(outdir, f"params_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(p.params, f, indent=2, ensure_ascii=False)
    print(p.describe())
    return theta, history


def main():
    ap = argparse.ArgumentParser(description="상수 최적화 BT (ES)")
    ap.add_argument("--generations", type=int, default=300)
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.10)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--outdir", default="results/es_bto")
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--fitness", default="default", choices=("default", "kill_first"))
    a = ap.parse_args()
    train(a.generations, a.pop, a.sigma, a.lr, a.episodes, a.seed, a.workers,
          outdir=a.outdir, checkpoint_every=a.checkpoint_every,
          tag=a.tag or f"seed{a.seed}", fitness=a.fitness)


if __name__ == "__main__":
    main()
