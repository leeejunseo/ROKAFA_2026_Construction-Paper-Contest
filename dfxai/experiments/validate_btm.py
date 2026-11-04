"""
BT-M(교범 기반 기준선 행동트리) G1 검증.

    python -m dfxai.experiments.validate_btm --variant core --workers 4   # BT-M  (기준선)
    python -m dfxai.experiments.validate_btm --variant full --workers 4   # BT-M+ (상세판)
    python -m dfxai.experiments.validate_btm --variant core --skip-6dof   # 3자유도만

G1 기준 (C안 실행 명세서)
    E1 3자유도·6자유도 각 400교전(시드 10000~10099 × 진영 교대 × 상대 BT-v2·BT-v3) 오류 없이 완료
    E2 공중충돌 ≤ 10 %
    E3 지면충돌(자기 추락) ≤ 5 %
    E4 BT-v2 상대 점수 ≥ 0.45 (3자유도·6자유도 각각)
    E5 행동복제 재현성 R² ≥ 0.9 (MLP 20-32-32-3, 3자유도 로그)
참고로 함께 보고: 상대별 점수, 결과 유형, 규칙 위반율, 노드 발동 빈도, D*(0.90/0.95/0.99), F(4).

E5 는 torch 가 있으면 dfxai.rl.pretrain_bc.fit(본 실험과 같은 학습기)을 쓰고,
없으면 같은 구조의 sklearn MLPRegressor(tanh, 32-32)로 대신한다(보고서에 어느 쪽인지 적힌다).
"""
from __future__ import annotations
import argparse, collections, json, os, time, traceback
from multiprocessing import Pool

import numpy as np

from ..env import run_episode
from ..agents.bt import BTPolicy
from ..agents.bt_manual import ManualBTPolicy, BTMParams

VARIANTS = {"core": BTMParams.core, "full": BTMParams.full}


def make_policy(variant: str) -> ManualBTPolicy:
    return ManualBTPolicy(VARIANTS[variant]())
from .run_eval import make_env

SEED0, N_SEEDS, OPPONENTS, N_RECORD = 10_000, 100, (2, 3), 30


def _job(args):
    opp, seed, side, sixdof, record, variant = args
    try:
        pol, o = make_policy(variant), BTPolicy(version=opp)
        env = make_env({"sixdof": True} if sixdof else {})
        nodes = collections.Counter()

        class Tap:                                   # 노드 발동 빈도 기록용 래퍼
            name = "BT-M"
            def act(self, obs):
                a = pol.act(obs); nodes[pol.last_node] += 1; return a

        tap = Tap()
        if side == 0:
            out = run_episode(tap, o, seed=seed, env=env, record=record)
            res, ob, ac = out if record else (out, None, None)
            win, v = res.winner, res.violations_blue
        else:
            res = run_episode(o, tap, seed=seed, env=env)
            ob = ac = None
            win, v = -res.winner, res.violations_red
        crashed_self = res.outcome == "crash" and win == -1
        row = dict(opp=f"BT-v{opp}", seed=seed, side=side, ok=True,
                   score=1.0 if win == 1 else 0.5 if win == 0 else 0.0, win=win,
                   outcome=res.outcome, crash_self=crashed_self, duration=res.duration,
                   viol_deck=v["deck"], viol_over_g=v["over_g"], viol_sep=v["separation"],
                   trim_ok=getattr(env, "trim_ok", True))
        return row, dict(nodes), (ob, ac)
    except Exception:                                # E1: 오류도 결과로 남긴다
        return dict(opp=f"BT-v{opp}", seed=seed, side=side, ok=False,
                    error=traceback.format_exc()[-400:]), {}, (None, None)


def evaluate(sixdof: bool, workers: int, variant: str = "core") -> dict:
    jobs = [(o, SEED0 + i, s, sixdof, s == 0 and i < N_RECORD, variant)
            for o in OPPONENTS for i in range(N_SEEDS) for s in (0, 1)]
    t0 = time.time()
    if workers > 1:
        with Pool(workers) as p:
            out = p.map(_job, jobs, chunksize=2)
    else:
        out = [_job(j) for j in jobs]
    rows = [r for r, _, _ in out]
    nodes = collections.Counter()
    for _, n, _ in out:
        nodes.update(n)
    obs = [x[0] for _, _, x in out if x[0] is not None]
    act = [x[1] for _, _, x in out if x[1] is not None]
    return dict(rows=rows, nodes=dict(nodes), obs=np.concatenate(obs) if obs else None,
                act=np.concatenate(act) if act else None, seconds=time.time() - t0)


def summarize(ev: dict) -> dict:
    ok = [r for r in ev["rows"] if r["ok"]]
    s = dict(n_total=len(ev["rows"]), n_ok=len(ok), n_err=len(ev["rows"]) - len(ok),
             collision=float(np.mean([r["outcome"] == "collision" for r in ok])),
             crash_self=float(np.mean([r["crash_self"] for r in ok])),
             viol_deck=float(np.mean([r["viol_deck"] for r in ok])),
             viol_over_g=float(np.mean([r["viol_over_g"] for r in ok])),
             viol_sep=float(np.mean([r["viol_sep"] for r in ok])),
             trim_ok=float(np.mean([r["trim_ok"] for r in ok])), seconds=ev["seconds"])
    for o in OPPONENTS:
        rr = [r for r in ok if r["opp"] == f"BT-v{o}"]
        sc = np.array([r["score"] for r in rr])
        s[f"score_v{o}"] = float(sc.mean())
        s[f"score_v{o}_ci"] = _wilson_like(sc)
        s[f"outcomes_v{o}"] = dict(collections.Counter(
            f"{r['outcome']}:{'승' if r['win'] == 1 else '무' if r['win'] == 0 else '패'}" for r in rr))
    tot = sum(ev["nodes"].values()) or 1
    s["node_freq"] = {k: v / tot for k, v in sorted(ev["nodes"].items(), key=lambda kv: -kv[1])}
    return s


def _wilson_like(sc: np.ndarray, n_boot: int = 2000) -> list:
    """점수(0/0.5/1) 평균의 95% 부트스트랩 구간."""
    rng = np.random.default_rng(0)
    m = [sc[rng.integers(0, len(sc), len(sc))].mean() for _ in range(n_boot)]
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def bc_reproducibility(n_episodes: int = 300, seed: int = 0, variant: str = "core") -> dict:
    """BT-M 행동을 MLP 20-32-32-3 으로 복제했을 때의 검증 R² (pretrain_bc.collect 와 같은 수집 방식)."""
    from ..env import DogfightEnv
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for i in range(n_episodes):
        ov = int(rng.choice((1, 2, 3)))
        _, ob, ac = run_episode(make_policy(variant), BTPolicy(version=ov), seed=200_000 + i,
                                alpha=float(rng.uniform()), env=DogfightEnv(), record=True)
        X.append(ob); Y.append(ac)
    X, Y = np.concatenate(X), np.concatenate(Y)
    X[:, -1] = np.random.default_rng(seed + 1).uniform(0.0, 1.0, X.shape[0])
    idx = np.random.default_rng(seed).permutation(len(X))
    n_val = max(1000, len(X) // 10)
    va, tr = idx[:n_val], idx[n_val:]
    try:
        from ..rl.pretrain_bc import fit
        pol = fit(X[tr], Y[tr], seed=seed, verbose=False)
        pred = np.stack([pol.act(x) for x in X[va]]); learner = "torch (dfxai.rl.pretrain_bc.fit)"
    except ImportError:
        from sklearn.neural_network import MLPRegressor
        m = MLPRegressor(hidden_layer_sizes=(32, 32), activation="tanh", max_iter=1000, tol=1e-6,
                         n_iter_no_change=50, learning_rate_init=2e-3, batch_size=512, random_state=seed)
        m.fit(X[tr], Y[tr])
        pred = np.clip(m.predict(X[va]), -1, 1); learner = f"sklearn MLPRegressor(tanh 32-32, {m.n_iter_}회 반복) — torch 없음"
    var = Y[va].var(axis=0)
    r2 = 1.0 - ((pred - Y[va]) ** 2).mean(axis=0) / np.maximum(var, 1e-9)
    w = var / var.sum()
    return dict(n_samples=int(len(X)), r2_per_channel=[float(v) for v in r2],
                r2_weighted=float(np.dot(w, r2)), learner=learner)


def surrogate(ev: dict) -> dict:
    from ..xai.surrogate import surrogate_fidelity
    r = surrogate_fidelity(ev["obs"], ev["act"])
    return dict(d_star=r.min_depth_at_target, d_star_multi=r.d_star_multi,
                f4=r.fidelity_at_fixed_depth, n_samples=r.n_samples)


def judge(s3: dict, s6: dict | None, bc: dict) -> list:
    out = []
    for tag, s in (("3자유도", s3), ("6자유도", s6)):
        if s is None:
            continue
        out += [
            (f"E1 {tag} 400교전 오류 없음", f"{s['n_ok']}/{s['n_total']}", s["n_err"] == 0 and s["n_total"] == 400),
            (f"E2 {tag} 공중충돌 ≤ 10%", f"{s['collision']:.3f}", s["collision"] <= 0.10),
            (f"E3 {tag} 지면충돌 ≤ 5%", f"{s['crash_self']:.3f}", s["crash_self"] <= 0.05),
            (f"E4 {tag} BT-v2 상대 점수 ≥ 0.45", f"{s['score_v2']:.3f} [{s['score_v2_ci'][0]:.3f}, {s['score_v2_ci'][1]:.3f}]",
             s["score_v2"] >= 0.45),
        ]
    out.append(("E5 행동복제 R² ≥ 0.9", f"{bc['r2_weighted']:.3f} (채널별 {', '.join(f'{v:.3f}' for v in bc['r2_per_channel'])})",
                bc["r2_weighted"] >= 0.9))
    return out


def report(s3, s6, bc, sur3, sur6, verdict, path, variant="core"):
    pol = make_policy(variant)
    P = pol.P.as_dict()
    L = [f"# {pol.name}(교범 기반 행동트리) G1 검증 결과 (자동 생성)", "",
         f"노드 {pol.complexity()['bt_nodes']}개(안전 하위트리 {pol.complexity()['safety_nodes']}개), 자율 G 제한 {pol.g_cap:.1f} G. "
         "평가: 시드 10000–10099 × 진영 교대 × 상대 BT-v2·BT-v3 = 환경당 400교전.", "",
         "## 1. G1 판정", "", "| 기준 | 값 | 판정 |", "|---|---|---|"]
    L += [f"| {a} | {b} | {'충족' if c else '미충족'} |" for a, b, c in verdict]
    L += ["", "## 2. 환경별 요약", "", "| 지표 | 3자유도 | 6자유도 |", "|---|---|---|"]
    keys = [("score_v2", "점수 vs BT-v2"), ("score_v3", "점수 vs BT-v3"), ("collision", "공중충돌률"),
            ("crash_self", "자기 추락률"), ("viol_deck", "하드덱 위반율(스텝)"), ("viol_over_g", "과G 위반율(스텝)"),
            ("viol_sep", "최소이격 위반율(스텝)"), ("trim_ok", "6자유도 트림 성공률"), ("seconds", "실행 시간 [s]")]
    for k, name in keys:
        a = s3.get(k); b = s6.get(k) if s6 else None
        f = (lambda x: "-" if x is None else f"{x:.4f}")
        L.append(f"| {name} | {f(a)} | {f(b)} |")
    L += ["", "## 3. 결과 유형", ""]
    for tag, s in (("3자유도", s3), ("6자유도", s6)):
        if s:
            for o in OPPONENTS:
                L.append(f"- {tag} vs BT-v{o}: {s[f'outcomes_v{o}']}")
    L += ["", "## 4. 설명가능성(참고)", "", "| 환경 | D*(0.95) | D*(0.90/0.95/0.99) | F(4) | 샘플 |", "|---|---|---|---|---|"]
    for tag, r in (("3자유도", sur3), ("6자유도", sur6)):
        if r:
            L.append(f"| {tag} | {r['d_star']:.0f} | {r['d_star_multi']} | {r['f4']:.3f} | {r['n_samples']} |")
    L += ["", f"행동복제 학습기: {bc['learner']}, 샘플 {bc['n_samples']:,}개.", "",
          "## 5. 노드 발동 빈도", "", "| 노드 | 3자유도 | 6자유도 |", "|---|---|---|"]
    names = list(s3["node_freq"].keys()) + [k for k in (s6 or {}).get("node_freq", {}) if k not in s3["node_freq"]]
    for k in names:
        b = (s6 or {}).get("node_freq", {}).get(k)
        L.append(f"| {k} | {s3['node_freq'].get(k, 0):.3f} | {'-' if b is None else f'{b:.3f}'} |")
    L += ["", "## 6. 상수", "", "```json", json.dumps(P, ensure_ascii=False, indent=1), "```", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="BT-M G1 검증")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--skip-6dof", action="store_true")
    ap.add_argument("--variant", choices=tuple(VARIANTS), default="core")
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    a.outdir = a.outdir or f"results/btm_g1/{make_policy(a.variant).name}"
    os.makedirs(a.outdir, exist_ok=True)
    e3 = evaluate(False, a.workers, a.variant); s3 = summarize(e3); sur3 = surrogate(e3)
    print("3자유도 완료", {k: s3[k] for k in ("score_v2", "score_v3", "collision", "crash_self")}, flush=True)
    s6 = sur6 = None
    if not a.skip_6dof:
        e6 = evaluate(True, a.workers, a.variant); s6 = summarize(e6); sur6 = surrogate(e6)
        print("6자유도 완료", {k: s6[k] for k in ("score_v2", "score_v3", "collision", "crash_self")}, flush=True)
    bc = bc_reproducibility(variant=a.variant)
    print("행동복제", bc, flush=True)
    v = judge(s3, s6, bc)
    with open(os.path.join(a.outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(dict(s3=s3, s6=s6, bc=bc, sur3=sur3, sur6=sur6,
                       verdict=[(x, y, bool(z)) for x, y, z in v]), f, ensure_ascii=False, indent=1)
    print(report(s3, s6, bc, sur3, sur6, v, os.path.join(a.outdir, "G1_report.md"), a.variant))


if __name__ == "__main__":
    main()
