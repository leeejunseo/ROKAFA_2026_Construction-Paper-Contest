"""
행동 복제(behavior cloning) 사전학습 — ES 미세조정의 출발점.

    python -m dfxai.rl.pretrain_bc --teacher 2 --out results/es/bc_init.npz

왜 필요한가
-----------
무작위 초기화에서 ES 를 돌리면 정책은 예외 없이 **원거리 회피**로 수렴했습니다
(README 5.5). 교전을 시도하면 BT-v2 에게 격추당해 적합도 -30, 도망가면 -9 이므로
탐색이 싸우는 정책을 발견할 확률이 사실상 0 입니다. 희소·고분산 보상에서
진화전략의 잘 알려진 한계입니다.

그래서 BT-v2 가 남긴 (관측, 행동) 로그에 MLP 를 회귀시켜 **BT 의 행동을 복제한
가중치**로 ES 를 시작합니다. 이후 학습 예산(세대)이 늘수록 정책은 복제본에서
멀어지며, 논문의 질문은 "그 과정에서 성능과 설명가능성이 어떻게 함께 움직이는가"
가 됩니다. alpha=1.0 정책은 여전히 MLP 만 실행하는 블랙박스입니다 — 학습 정책의
정의는 유지되고, 초기화 방법만 명시하면 됩니다(논문 방법 절에 반드시 쓸 것).

세부
----
* 교사: BT-v2 (학습 상대와 같은 계열). 상대는 BT-v1/v2/v3 를 섞어 다양한 기하를 봅니다.
* 관측의 alpha 열은 무작위 [0,1] 로 채웁니다 — 학생이 alpha 에 무감각해지도록.
* 손실: MSE. 뱅크 채널은 ±1 경계(=±180°)에서 불연속이므로 sin/cos 로 바꿔
  회귀하고 싶지만, 정책 출력 형식을 유지하기 위해 그대로 둡니다. 복제 정확도는
  R^2 로 보고하며, 완벽할 필요는 없습니다(ES 가 이어서 다듬습니다).
* torch 로 학습하고 MLPPolicy 포맷(flat, hidden)으로 저장합니다.

C안(2026-11, E4): 교사·상대를 이름으로 고른다(--teacher btm, registry). --sixdof 이면 6자유도 교전에서
(관측, 행동)을 모으고, --workers 로 나눠 모은다(교전마다 상대·α 를 미리 뽑아 두므로 작업자 수와 무관하게
같은 데이터). --save-data 로 수집 자료를 따로 남기고, 복제 정확도는 <out>.json 에 적는다.
아무것도 안 주면 기존과 같은 3자유도 BT-v2 복제다.
"""
from __future__ import annotations
import argparse, os, time
import numpy as np

from ..env import run_episode
from ..agents.registry import make_bt, label
from ..agents.hybrid import MLPPolicy
from ..config import OBS_DIM, ACTION_DIM
from .es_runtime import make_env, env_opts as _env_opts


def _collect_one(args):
    teacher, ov, seed, alpha, opts = args
    _, obs, act = run_episode(make_bt(teacher), make_bt(ov), seed=seed, alpha=alpha,
                              env=make_env(opts), record=True)
    return obs, act


def collect(teacher=2, opponents=(1, 2, 3), n_episodes: int = 300,
            seed0: int = 200_000, rng_seed: int = 0, env_opts: dict | None = None,
            workers: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """교사 BT 의 (관측, 행동) 로그. 진영을 번갈아 청/홍 양쪽 관점을 모읍니다."""
    rng = np.random.default_rng(rng_seed)
    jobs = []
    for i in range(n_episodes):
        ov = opponents[int(rng.choice(len(opponents)))]     # rng.choice(opponents) 와 같은 난수 소비
        alpha = float(rng.uniform(0.0, 1.0))
        jobs.append((teacher, ov, seed0 + i, alpha, env_opts))
    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as p:
            out = p.map(_collect_one, jobs, chunksize=2)
    else:
        out = [_collect_one(j) for j in jobs]
    X = np.concatenate([o for o, _ in out]); Y = np.concatenate([a for _, a in out])
    X[:, -1] = np.random.default_rng(rng_seed + 1).uniform(0.0, 1.0, X.shape[0])
    return X, Y


def fit(X: np.ndarray, Y: np.ndarray, hidden=(32, 32), epochs: int = 60,
        lr: float = 2e-3, batch: int = 512, seed: int = 0, verbose: bool = True) -> MLPPolicy:
    import torch
    torch.manual_seed(seed)
    n = X.shape[0]
    idx = np.random.default_rng(seed).permutation(n)
    n_val = max(1000, n // 10)
    va, tr = idx[:n_val], idx[n_val:]
    Xt = torch.tensor(X, dtype=torch.float32); Yt = torch.tensor(Y, dtype=torch.float32)

    sizes = (OBS_DIM,) + tuple(hidden) + (ACTION_DIM,)
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [torch.nn.Linear(a, b), torch.nn.Tanh()]
    net = torch.nn.Sequential(*layers)              # 마지막 Tanh 는 MLPPolicy 와 동일
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    t0 = time.time()
    for ep in range(1, epochs + 1):
        perm = torch.tensor(np.random.default_rng(seed + ep).permutation(tr))
        net.train()
        for s in range(0, len(perm), batch):
            b = perm[s:s + batch]
            loss = torch.mean((net(Xt[b]) - Yt[b]) ** 2)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if verbose and (ep % 10 == 0 or ep == 1):
            net.eval()
            with torch.no_grad():
                pv = net(Xt[va]).numpy()
            r2 = 1.0 - ((pv - Y[va]) ** 2).mean(axis=0) / (Y[va].var(axis=0) + 1e-9)
            print(f"[bc ep {ep:3d}] val R2 per channel = {np.round(r2, 3)}  ({time.time()-t0:.0f}s)", flush=True)
    net.eval()
    with torch.no_grad():
        pv = net(Xt[va]).numpy()
    sse, var = ((pv - Y[va]) ** 2).mean(axis=0), Y[va].var(axis=0)
    # 분산 가중 R² (D* 와 같은 정의): 상수 채널(예: 스로틀 고정)은 분모에 거의 기여하지 않는다
    r2_ch = [float(1.0 - e / v) if v > 1e-6 else None for e, v in zip(sse, var)]
    r2_w = float(1.0 - sse.sum() / max(var.sum(), 1e-12))

    # torch Linear 는 y = x W^T + b. MLPPolicy 는 x @ W + b 이므로 전치해서 평탄화.
    flat = []
    for m in net:
        if isinstance(m, torch.nn.Linear):
            flat.append(m.weight.detach().numpy().T.reshape(-1))
            flat.append(m.bias.detach().numpy().reshape(-1))
    pol = MLPPolicy(hidden=tuple(hidden), params=np.concatenate(flat).astype(np.float64))
    # 이식 검증
    with torch.no_grad():
        ref = net(Xt[va[:64]]).numpy()
    mine = np.stack([pol.act(x) for x in X[va[:64]]])
    assert np.abs(ref - mine).max() < 1e-4, "torch -> MLPPolicy 가중치 이식 불일치"
    pol.bc_r2 = dict(weighted=r2_w, per_channel=r2_ch, n_train=int(len(tr)), n_val=int(len(va)))
    return pol


def evaluate(pol: MLPPolicy, opponent=2, n_seeds: int = 30, seed0: int = 60_000,
             env_opts: dict | None = None) -> dict:
    """복제본이 실제로 싸우는지 확인 (진영 교대)."""
    import collections
    oc = collections.Counter(); sc = []; wez = []
    env = make_env(env_opts)
    for s in range(seed0, seed0 + n_seeds):
        r = run_episode(pol, make_bt(opponent), seed=s, alpha=1.0, env=env)
        sc.append(1.0 if r.winner == 1 else 0.5 if r.winner == 0 else 0.0); oc[r.outcome] += 1; wez.append(r.wez_time_blue)
        r = run_episode(make_bt(opponent), pol, seed=s, alpha=0.0, alpha_red=1.0, env=env)
        sc.append(1.0 if r.winner == -1 else 0.5 if r.winner == 0 else 0.0); oc[r.outcome] += 1; wez.append(r.wez_time_red)
    return dict(score=float(np.mean(sc)), wez=float(np.mean(wez)), outcomes=dict(oc))


def main():
    ap = argparse.ArgumentParser(description="BT 행동 복제 사전학습")
    ap.add_argument("--teacher", default="2", help="교사 행동트리 (정수 1–3 또는 이름 btm, btmplus, school)")
    ap.add_argument("--opponents", nargs="*", default=["1", "2", "3"], help="수집 교전의 상대 행동트리")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", default="results/es/bc_init.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sixdof", action="store_true", help="6자유도 교전에서 수집·평가")
    ap.add_argument("--cone", nargs=2, type=float, default=None, metavar=("START", "END"),
                    help="기총 명중 원추 반각 [deg] (기본: 3 → 10)")
    ap.add_argument("--roll-rate-max", type=float, default=None, help="3자유도 롤 속도 상한 [deg/s]")
    ap.add_argument("--workers", type=int, default=1, help="수집 병렬 작업자 수")
    ap.add_argument("--save-data", default=None, help="수집한 (관측, 행동)을 이 npz 로 저장")
    ap.add_argument("--eval-opponents", nargs="*", default=["2", "3"], help="복제본 점검 상대")
    a = ap.parse_args()
    import json, sys
    sys.stdout.reconfigure(encoding="utf-8")
    num = lambda v: int(v) if str(v).isdigit() else v          # 정수 이름은 기존과 같은 난수 소비
    teacher, opps = num(a.teacher), tuple(num(o) for o in a.opponents)
    opts = _env_opts(a.sixdof, a.roll_rate_max, a.cone)
    t0 = time.time()
    X, Y = collect(teacher, opps, n_episodes=a.episodes, rng_seed=a.seed, env_opts=opts, workers=a.workers)
    print(f"로그 수집: {X.shape[0]:,} 샘플 ({time.time()-t0:.0f}s)", flush=True)
    if a.save_data:
        os.makedirs(os.path.dirname(os.path.abspath(a.save_data)) or ".", exist_ok=True)
        np.savez_compressed(a.save_data, obs=X, act=Y, teacher=str(teacher), sixdof=bool(a.sixdof))
    pol = fit(X, Y, epochs=a.epochs, seed=a.seed)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    pol.save(a.out)
    rep = dict(teacher=label(teacher), opponents=[label(o) for o in opps], sixdof=bool(a.sixdof),
               episodes=a.episodes, samples=int(X.shape[0]), r2=pol.bc_r2, eval={})
    print(f"복제 정확도(검증 자료): 분산 가중 R² {pol.bc_r2['weighted']:.3f}, 채널별 {pol.bc_r2['per_channel']}")
    for ov in a.eval_opponents:
        r = evaluate(pol, num(ov), env_opts=opts)
        rep["eval"][label(ov)] = r
        print(f"복제본 vs {label(ov)}:", r, flush=True)
    with open(os.path.splitext(a.out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(f"저장: {a.out}  (교사 {label(teacher)})")


if __name__ == "__main__":
    main()
