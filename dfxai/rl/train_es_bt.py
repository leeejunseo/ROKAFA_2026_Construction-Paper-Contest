"""
상수 최적화 행동트리(BTO) 학습기.

bt_param.ParamBTPolicy 의 상수 9개를 학습 정책과 **같은 적합도·같은 예산**
(300세대 × 개체군 32 × 후보당 8교전, BT-v1/v2 상대 교대, 갱신 수용 검사)으로
진화전략 최적화한다. 트리 구조와 노드 수는 고정이므로 결과물은 여전히 사람이
읽을 수 있는 규칙이다.

학습 정책과 다른 점은 탐색 파라미터뿐이다. 신경망은 1,827차원 가중치 공간이라
σ 0.05·lr 0.01 을 썼고, BTO 는 [-1, 1] 로 정규화한 9차원이라 σ 0.10·lr 0.05 를
쓴다(차원이 다르면 같은 보폭이 같은 의미가 아니다). 이 차이는 논문에 명시한다.

C안(2026-11): --bt 로 상수를 공개한 행동트리를 고른다(registry 의 PARAM_SPEC). 기본 'v3' 는 기존
bt_param.py 와 같은 정책이다. 체크포인트에 행동트리 이름을 함께 저장하고 load_tunable() 로 읽는다.
--opponents·--sixdof·--resume 는 train_es.py 와 같다. 아무것도 안 주면 기존 학습과 똑같다.
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
from multiprocessing import Pool

from ..agents.registry import make_bt, canon, param_spec, default_theta
from .shaping import potential, episode_fitness
from .train_es import _rank_transform, _dump_history
from .es_runtime import make_env, env_opts, save_state, load_state, write_meta, fresh_check, pmap


def save_tunable(pol, bt_name: str, path: str) -> None:
    np.savez(path, theta=np.asarray(pol.theta, float), kind=np.array("bto"), bt_name=np.array(canon(bt_name)))


def load_tunable(path: str):
    """파라미터 최적화형 체크포인트 -> 정책. 옛 파일(bt_name 없음)은 BT-v3 구조(bt_param.py)."""
    d = np.load(path)
    return make_bt(str(d["bt_name"]) if "bt_name" in d.files else "v3", theta=d["theta"])


def _rollout(args):
    theta, seeds, opps, preset, *rest = args
    bt_name = rest[0] if rest else "v3"
    opts = rest[1] if len(rest) > 1 else None
    pol = make_bt(bt_name, theta=theta)
    env = make_env(opts)
    total = 0.0
    for seed, ov in zip(seeds, opps):
        red = make_bt(ov)
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
          checkpoint_every=50, tag="seed0", fitness="default", bt="v3", env_opts=None,
          resume=False, stop_after=None, meta_args=None, accept_fresh=False, accept_episodes=None):
    """bt: 상수를 공개한 행동트리 이름(기본 v3 = bt_param.py). 나머지 C안 인자는 train_es.train 과 같다."""
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_par = len(param_spec(bt))
    theta = default_theta(bt) + rng.normal(0.0, 0.05, n_par)   # 기본 상수 근방에서 출발
    lr_eff, n_accept, history = lr, 0, []
    start = 1
    st = load_state(outdir, tag, rng) if resume else None
    if st:
        start, theta, lr_eff, n_accept, history = st
        start += 1
        print(f"이어하기: {start - 1}세대까지 끝난 상태에서 시작 (수용 {n_accept}, lr {lr_eff:.4f})", flush=True)
    write_meta(outdir, tag, meta_args or {}, gen=start - 1)
    pool = Pool(workers) if workers > 1 else None
    t0 = time.time() - (history[-1]["elapsed"] if history else 0.0)
    for gen in range(start, generations + 1):
        seeds = rng.integers(0, 10 ** 6, episodes)
        opps = np.array([opponents[i % len(opponents)] for i in range(episodes)])
        rng.shuffle(opps)
        eps = rng.normal(0.0, 1.0, (pop // 2, n_par))
        eps = np.concatenate([eps, -eps], axis=0)
        cands = np.clip(theta[None, :] + sigma * eps, -1.0, 1.0)
        jobs = [(cands[i], seeds, opps, fitness, bt, env_opts) for i in range(pop)]
        fits = (np.array(pool.map(_rollout, jobs)) if pool
                else np.array([_rollout(j) for j in jobs]))
        grad = (_rank_transform(fits)[:, None] * eps).mean(axis=0) / sigma
        theta_new = np.clip(theta + lr_eff * grad, -1.0, 1.0)
        if accept_fresh:                     # C안: 기울기와 다른 시드로 수용 판정 (train_es.py 참고)
            ne = accept_episodes or episodes
            c_seeds, c_opps = fresh_check(rng, opponents, ne)
            r = pmap(pool, _rollout, [(th, c_seeds[i:i + 1], c_opps[i:i + 1], fitness, bt, env_opts)
                                      for th in (theta, theta_new) for i in range(ne)])
            f_old, f_new = float(np.mean(r[:ne])), float(np.mean(r[ne:]))
        else:
            chk = [(theta, seeds, opps, fitness, bt, env_opts), (theta_new, seeds, opps, fitness, bt, env_opts)]
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
            save_tunable(make_bt(bt, theta=theta), bt, os.path.join(outdir, f"ckpt_{tag}_gen{gen:05d}.npz"))
            _dump_history(outdir, tag, history)
        save_state(outdir, tag, gen, theta, rng, lr_eff, n_accept, history)
        if stop_after and gen - start + 1 >= stop_after and gen < generations:
            print(f"--stop-after {stop_after}: {gen}세대에서 멈춤 (--resume 으로 이어서)", flush=True)
            if pool:
                pool.close(); pool.join()
            return theta, history
    if pool:
        pool.close(); pool.join()
    if history and not history[-1].get("final"):
        history[-1]["final"] = True
        _dump_history(outdir, tag, history)
    p = make_bt(bt, theta=theta)
    params = getattr(p, "params", None) or {k: float(v) for k, v in zip((n for n, *_ in param_spec(bt)), theta)}
    with open(os.path.join(outdir, f"params_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    print(p.describe() if hasattr(p, "describe") else params)
    write_meta(outdir, tag, meta_args or {}, finished=True, gen=generations)
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
    ap.add_argument("--bt", default="v3", help="상수를 공개한 행동트리 이름 (기본 v3 = bt_param.py)")
    ap.add_argument("--opponents", nargs="*", default=["1", "2"], help="학습 상대 행동트리(정수 또는 이름)")
    ap.add_argument("--sixdof", action="store_true", help="6자유도(JSBSim F-16)에서 학습")
    ap.add_argument("--cone", nargs=2, type=float, default=None, metavar=("START", "END"),
                    help="기총 명중 원추 반각 [deg] (기본: 3 → 10)")
    ap.add_argument("--roll-rate-max", type=float, default=None, help="3자유도 롤 속도 상한 [deg/s]")
    ap.add_argument("--resume", action="store_true", help="state_<tag> 가 있으면 거기서 이어서 학습")
    ap.add_argument("--stop-after", type=int, default=None, help="이번 실행에서 이 세대 수만 돌고 멈춤(시험용)")
    ap.add_argument("--accept-fresh", action="store_true", help="수용 검사를 기울기와 다른 새 시드로 (C안)")
    ap.add_argument("--accept-episodes", type=int, default=None, help="--accept-fresh 수용 검사 교전 수")
    a = ap.parse_args()
    train(a.generations, a.pop, a.sigma, a.lr, a.episodes, a.seed, a.workers,
          opponents=tuple(a.opponents), outdir=a.outdir, checkpoint_every=a.checkpoint_every,
          tag=a.tag or f"seed{a.seed}", fitness=a.fitness, bt=a.bt,
          env_opts=env_opts(a.sixdof, a.roll_rate_max, a.cone), resume=a.resume, stop_after=a.stop_after,
          meta_args=dict(script="train_es_bt", **vars(a)), accept_fresh=a.accept_fresh,
          accept_episodes=a.accept_episodes)


if __name__ == "__main__":
    main()
