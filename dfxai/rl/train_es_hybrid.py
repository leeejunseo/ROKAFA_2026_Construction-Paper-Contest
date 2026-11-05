"""
잔차형(RES)·게이팅형(GATE) 혼합 정책의 ES 학습기.

train_es_bt.py(BTO) 와 같은 방식이다: ES 루프(미러 샘플링·순위 정규화·갱신 수용
검사)는 그대로 두고 **학습 대상 파라미터 블록만** 바꾼다.

    --kind residual : ResidualPolicy 의 δ-신경망 1,827개 (agents/residual.py)
    --kind gating   : GatingPolicy 의 게이트 177개 (agents/gating.py). 학습 정책 절반은
                      --rl-ckpt 로 준 기존 순수 학습 정책 체크포인트를 고정해서 쓴다.

적합도·상대(BT-v1/v2 교대)·개체군 32·후보당 8교전은 순수 학습 정책과 같다. 관측의
α 원소는 두 계열 모두 1.0 으로 고정한다(혼합 비율은 정책 안에서 정해지므로).

    python -m dfxai.rl.train_es_hybrid --kind residual --beta 0.35 --seed 0 --generations 200
    python -m dfxai.rl.train_es_hybrid --kind gating --rl-ckpt results/es/ckpt_seed11_gen00300.npz --seed 0

C안(2026-11): --bt 로 기준선 행동트리를 이름으로 고르고(registry), --opponents·--sixdof·--resume 는
train_es.py 와 같다. 아무것도 안 주면 기존 3자유도 학습과 똑같다.

학습 중 진단: 매 세대 현재 정책을 같은 시드로 평가하면서 잔차형은 |δ| 평균을,
게이팅형은 g 의 평균·표준편차를 history 에 남긴다. 잔차형의 |δ| 가 0 에 붙거나
게이팅형의 g 표준편차가 0 에 붙으면 "학습은 되는데 의도한 대로 안 되는" 경우다
(하이브리드_확장_실험설계.md 7절).
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
from multiprocessing import Pool

from ..agents.registry import make_bt
from ..agents.hybrid import MLPPolicy
from ..agents.residual import ResidualPolicy
from ..agents.gating import GatingPolicy
from .shaping import potential, episode_fitness
from .train_es import _rank_transform, _dump_history
from .es_runtime import make_env, env_opts, save_state, load_state, write_meta, fresh_check, pmap


def _make(kind: str, theta: np.ndarray, cfg: dict):
    if kind == "residual":
        return ResidualPolicy(beta=cfg["beta"], bt_version=cfg["bt_version"],
                              hidden=cfg["hidden"], params=theta)
    rl = MLPPolicy(hidden=cfg["rl_hidden"], params=cfg["rl_flat"], out_act=cfg["rl_out_act"])
    return GatingPolicy(rl, bt_version=cfg["bt_version"], gate_hidden=cfg["gate_hidden"],
                        gate_params=theta)


def _rollout(args):
    kind, theta, cfg, seeds, opps, preset, diag = args
    pol = _make(kind, theta, cfg)
    env = make_env(cfg.get("env_opts"))
    total, stat = 0.0, []
    for seed, ov in zip(seeds, opps):
        red = make_bt(ov)
        ob, orr = env.reset(seed=int(seed), alpha=1.0)
        pot_sum, n = 0.0, 0
        for _ in range(env.max_steps + 1):
            a_b = pol.act(ob)
            if diag:
                stat.append(float(np.abs(pol.last_delta).mean()) if kind == "residual"
                            else float(pol.last_gate))
            ob, orr, done = env.step(a_b, red.act(orr))
            pot_sum += potential(ob); n += 1
            if done:
                break
        total += episode_fitness(env.result(), pot_sum / max(n, 1), preset)
    if diag:
        s = np.asarray(stat)
        return total / len(seeds), (float(s.mean()), float(s.std()))
    return total / len(seeds)


def train(kind: str, generations=200, pop=32, sigma=0.05, lr=0.01, episodes=8, seed=0,
          workers=1, opponents=(1, 2), outdir="results/es_hybrid", checkpoint_every=50,
          tag="seed0", fitness="default", beta=0.35, bt_version=2, rl_ckpt=None,
          gate_hidden=(8,), env_opts=None, resume=False, stop_after=None, meta_args=None,
          accept_fresh=False, accept_episodes=None):
    """bt_version: 기준선 행동트리(정수 또는 registry 이름). 나머지 C안 인자는 train_es.train 과 같다."""
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)
    if kind == "residual":
        proto = ResidualPolicy(beta=beta, bt_version=bt_version, seed=seed)
        cfg = dict(beta=beta, bt_version=bt_version, hidden=proto.mlp.hidden)
    else:
        assert rl_ckpt, "--kind gating 은 --rl-ckpt 가 필요합니다"
        rl = MLPPolicy.load(rl_ckpt)
        proto = GatingPolicy(rl, bt_version=bt_version, gate_hidden=gate_hidden, seed=seed)
        cfg = dict(bt_version=bt_version, gate_hidden=tuple(gate_hidden), rl_hidden=rl.hidden,
                   rl_flat=rl.flat, rl_out_act=rl.out_act)
    cfg["env_opts"] = env_opts
    theta = proto.flat.copy()
    n_par = theta.size
    print(f"[{kind}] 학습 파라미터 {n_par}개, pop {pop}, 후보당 {episodes}교전, "
          f"σ {sigma}, lr {lr}" + (f", β {beta}" if kind == "residual" else f", rl {rl_ckpt}"),
          flush=True)
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
        cands = theta[None, :] + sigma * eps
        jobs = [(kind, cands[i], cfg, seeds, opps, fitness, False) for i in range(pop)]
        fits = (np.array(pool.map(_rollout, jobs)) if pool
                else np.array([_rollout(j) for j in jobs]))
        grad = (_rank_transform(fits)[:, None] * eps).mean(axis=0) / sigma
        theta_new = theta + lr_eff * grad
        if accept_fresh:                     # C안: 기울기와 다른 시드로 수용 판정 (train_es.py 참고)
            ne = accept_episodes or episodes
            c_seeds, c_opps = fresh_check(rng, opponents, ne)
            r = pmap(pool, _rollout, [(kind, th, cfg, c_seeds[i:i + 1], c_opps[i:i + 1], fitness, True)
                                      for th in (theta, theta_new) for i in range(ne)])

            def pooled(rs):                  # 교전별 (평균, 표준편차) → 전체 근사
                m = np.array([d[0] for _, d in rs]); s = np.array([d[1] for _, d in rs])
                return float(m.mean()), float(np.sqrt(np.mean(s ** 2) + m.var()))
            f_old, d_old = float(np.mean([f for f, _ in r[:ne]])), pooled(r[:ne])
            f_new, d_new = float(np.mean([f for f, _ in r[ne:]])), pooled(r[ne:])
        else:
            chk = [(kind, theta, cfg, seeds, opps, fitness, True),
                   (kind, theta_new, cfg, seeds, opps, fitness, True)]
            (f_old, d_old), (f_new, d_new) = (pool.map(_rollout, chk) if pool
                                              else [_rollout(j) for j in chk])
        accepted = f_new >= f_old
        if accepted:
            theta = theta_new; lr_eff = min(lr, lr_eff * 1.15)
        else:
            lr_eff = max(lr * 0.25, lr_eff * 0.7)
        n_accept += int(accepted)
        d_cur = d_new if accepted else d_old
        history.append(dict(gen=gen, fit_mean=float(fits.mean()), fit_max=float(fits.max()),
                            fit_theta=float(f_new if accepted else f_old),
                            accepted=bool(accepted), lr_eff=float(lr_eff),
                            elapsed=time.time() - t0, alpha_lo=1.0,
                            diag_mean=d_cur[0], diag_std=d_cur[1]))
        if gen % max(1, generations // 20) == 0 or gen == 1:
            lab = "|δ|" if kind == "residual" else "g"
            print(f"[gen {gen:4d}] fit mean={fits.mean():8.3f} max={fits.max():8.3f} "
                  f"theta={history[-1]['fit_theta']:8.3f} accept={n_accept}/{gen} "
                  f"lr={lr_eff:.4f} {lab}={d_cur[0]:.3f}±{d_cur[1]:.3f} ({time.time()-t0:.0f}s)",
                  flush=True)
        if gen % checkpoint_every == 0 or gen == generations:
            _make(kind, theta, cfg).save(os.path.join(outdir, f"ckpt_{tag}_gen{gen:05d}.npz"))
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
    write_meta(outdir, tag, meta_args or {}, finished=True, gen=generations)
    return theta, history


def main():
    ap = argparse.ArgumentParser(description="잔차형·게이팅형 혼합 ES 학습기")
    ap.add_argument("--kind", required=True, choices=("residual", "gating"))
    ap.add_argument("--generations", type=int, default=200)
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--fitness", default="default", choices=("default", "kill_first"))
    ap.add_argument("--beta", type=float, default=0.35, help="잔차형 보정 허용폭")
    ap.add_argument("--bt-version", type=int, default=2)
    ap.add_argument("--bt", default=None, help="기준선 행동트리 이름(v1–v3, btm, btmplus, school). 주면 --bt-version 대신 씀")
    ap.add_argument("--opponents", nargs="*", default=["1", "2"], help="학습 상대 행동트리(정수 또는 이름)")
    ap.add_argument("--sixdof", action="store_true", help="6자유도(JSBSim F-16)에서 학습")
    ap.add_argument("--cone", nargs=2, type=float, default=None, metavar=("START", "END"),
                    help="기총 명중 원추 반각 [deg] (기본: 3 → 10)")
    ap.add_argument("--roll-rate-max", type=float, default=None, help="3자유도 롤 속도 상한 [deg/s]")
    ap.add_argument("--resume", action="store_true", help="state_<tag> 가 있으면 거기서 이어서 학습")
    ap.add_argument("--stop-after", type=int, default=None, help="이번 실행에서 이 세대 수만 돌고 멈춤(시험용)")
    ap.add_argument("--rl-ckpt", default=None, help="게이팅형이 고정해서 쓸 학습 정책 체크포인트")
    ap.add_argument("--gate-hidden", type=int, nargs="*", default=[8])
    ap.add_argument("--accept-fresh", action="store_true", help="수용 검사를 기울기와 다른 새 시드로 (C안)")
    ap.add_argument("--accept-episodes", type=int, default=None, help="--accept-fresh 수용 검사 교전 수")
    a = ap.parse_args()
    outdir = a.outdir or ("results/es_res" if a.kind == "residual" else "results/es_gate")
    train(a.kind, a.generations, a.pop, a.sigma, a.lr, a.episodes, a.seed, a.workers,
          opponents=tuple(a.opponents),
          outdir=outdir, checkpoint_every=a.checkpoint_every, tag=a.tag or f"seed{a.seed}",
          fitness=a.fitness, beta=a.beta, bt_version=a.bt or a.bt_version, rl_ckpt=a.rl_ckpt,
          gate_hidden=tuple(a.gate_hidden), env_opts=env_opts(a.sixdof, a.roll_rate_max, a.cone),
          resume=a.resume, stop_after=a.stop_after, meta_args=dict(script="train_es_hybrid", **vars(a)),
          accept_fresh=a.accept_fresh, accept_episodes=a.accept_episodes)


if __name__ == "__main__":
    main()
