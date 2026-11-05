"""
기준선 BT 수용 검사 (C안 G1 = 손형래 H4, 이준서와 함께).

    python -m dfxai.cplan.accept_bt --bt btm --workers 11            (이 PC 약 30분)
    python -m dfxai.cplan.accept_bt --bt tmpl --quick --workers 11   (형식만 빠르게)

검사 항목
  형식 (H1·H3)  act() 가 [-1, 1]^3 유한값을 내고 last_node 를 남기는가, 기억이 없는가(같은 관측 = 같은 지령),
                SAFETY_NODES 가 실제 발동하는 노드인가, PARAM_SPEC 이 5–10개이고 기본값 θ 가 기본 정책과 같은가,
                rule_count() 가 있는가
  교전 상대는 plan.json 의 다른 학습 상대와 보류 상대, 환경 규칙(6자유도 기총 원추 등)도 plan.json 그대로다.
  ① 오류 없음              3DOF·6DOF 각각 상대마다 시드 100 × 진영 2
  ② 공중충돌 ≤ 10 %, 지면충돌(자기 추락) ≤ 5 %   (제안 기준, 3DOF·6DOF 각각)
  ③ 학습 상대끼리 점수 0.35–0.65 (어느 쪽도 압도하지 않음, 3DOF·6DOF 각각. 손형래 제안 2026-11-05)
  ④ 행동 복제 가능: 복제본의 지령 재현 R² ≥ 0.9 (분산 가중, 검증 자료, 3DOF·6DOF 각각, 기본 1000회 반복)
부산물: 행동 복제 초기값 results/cplan/bc/bc_<bt>_<3dof|6dof>.npz (+ .json, 수집 자료 _data.npz) — 초기화형 학습의 출발점(E4).
        수집 교전의 상대는 학습 상대만 쓴다(보류 상대는 행동 복제에도 쓰지 않는다).

출력: results/cplan/accept_<bt>.md, .json
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
import numpy as np
import pandas as pd
from multiprocessing import Pool

from ..agents.registry import make_bt, label, canon, param_spec, default_theta, safety_nodes
from ..agents.base import Policy
from ..env import DogfightEnv, run_episode
from ..experiments.run_eval import _one_job
from ..rl.es_runtime import plan_env_opts

TH = dict(collision=0.10, crash=0.05, partner_lo=0.35, partner_hi=0.65, bc_r2=0.90)


def load_plan() -> dict:
    """job.load_plan 과 같은 파일. job 을 불러오면 스레드 수가 1로 고정되어 행동 복제(torch)가 느려지므로 따로 읽는다."""
    path = os.environ.get("CPLAN_PLAN") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "plan.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _safe_job(args):
    try:
        return _one_job(args)[0]
    except Exception:
        spec, ov, seed, side = args[:4]
        return dict(cond=spec["name"], opponent=label(ov), seed=seed, side=side, error=traceback.format_exc())


def interface_checks(name: str) -> list[tuple]:
    """형식 점검. (항목, 통과 여부, 설명) 목록."""
    out = []
    pol = make_bt(name)
    out.append(("Policy 상속", isinstance(pol, Policy), type(pol).__name__))
    env = DogfightEnv()
    obs = []
    nodes = set()
    for seed in range(3):
        ob, orr = env.reset(seed=50_000 + seed)
        opp = make_bt(2)
        for _ in range(env.max_steps + 1):
            obs.append(ob.copy())
            a = pol.act(ob)
            nodes.add(getattr(pol, "last_node", ""))
            ob, orr, done = env.step(a, opp.act(orr))
            if done:
                break
    acts = np.array([make_bt(name).act(o) for o in obs])
    ok_shape = acts.ndim == 2 and acts.shape[1] == 3
    ok_range = bool(np.isfinite(acts).all() and np.abs(acts).max() <= 1.0 + 1e-12)
    out.append(("act() 출력 [-1, 1]^3 유한값", ok_shape and ok_range,
                f"범위 [{acts.min():.3f}, {acts.max():.3f}]" if ok_shape else f"모양 {acts.shape}"))
    out.append(("last_node 기록", "" not in nodes and len(nodes) > 0, f"발동 노드 {sorted(nodes)}"))
    fresh = make_bt(name)
    rev = np.array([fresh.act(o) for o in obs[::-1]])[::-1]
    out.append(("기억 없음 (같은 관측 = 같은 지령, 순서 무관)", bool(np.array_equal(acts, rev)),
                "D* 는 관측의 함수인 정책을 전제로 한다. 내부 상태가 있으면 실패"))
    sn = getattr(pol, "SAFETY_NODES", None)
    out.append(("SAFETY_NODES 공개", bool(sn), f"{sn}" if sn else "없으면 감독형이 BT-v3 의 노드 이름을 찾는다"))
    try:
        spec = param_spec(name)
        okn = 5 <= len(spec) <= 10
        okb = all(lo < hi and lo <= d <= hi for _, lo, hi, d in spec)
        out.append(("PARAM_SPEC 5–10개, 하한 ≤ 기본값 ≤ 상한", okn and okb, f"{len(spec)}개"))
        th0 = default_theta(name)
        a0 = np.array([make_bt(name, theta=th0).act(o) for o in obs[:400]])
        out.append(("기본값 θ 정책 = 기본 정책", bool(np.allclose(a0, acts[:400], atol=1e-9)),
                    f"최대 차이 {np.abs(a0 - acts[:400]).max():.2e}"))
        for edge in (-1.0, 1.0):
            p = make_bt(name, theta=np.full(len(spec), edge))
            e = np.array([p.act(o) for o in obs[:200]])
            out.append((f"θ = {edge:+.0f} (모든 상수 경계값) 동작", bool(np.isfinite(e).all() and np.abs(e).max() <= 1.0),
                        ""))
    except Exception as ex:
        out.append(("PARAM_SPEC (파라미터 최적화형용)", False, str(ex)))
    rc = getattr(pol, "rule_count", None)
    out.append(("rule_count()", callable(rc), f"{rc()}" if callable(rc) else "논문 3.2절에 보고할 규칙 수"))
    return out, nodes


def battle(name: str, opps: list, opts: dict, n_seeds: int, workers: int) -> pd.DataFrame:
    spec = dict(name=label(name), kind="bt", bt=canon(name))
    jobs = [(spec, ov, 10_000 + i, side, False, opts) for ov in opps for i in range(n_seeds) for side in (0, 1)]
    with Pool(workers) as p:
        rows = p.map(_safe_job, jobs, chunksize=4)
    return pd.DataFrame(rows)


def judge_battle(df: pd.DataFrame) -> dict:
    err = int(df["error"].notna().sum()) if "error" in df else 0
    ok = df[df["error"].isna()] if "error" in df else df
    return dict(n=len(df), errors=err, collision=float((ok.outcome == "collision").mean()),
                crash=float(((ok.outcome == "crash") & (ok.loss == 1)).mean()),
                score={o: float(g.score.mean()) for o, g in ok.groupby("opponent")},
                contact=float(((ok.damage_dealt + ok.damage_taken) > 0).mean()), wez=float(ok.wez_time.mean()),
                kill_win=float(((ok.outcome == "gun_kill") & (ok.win == 1)).mean()),
                timeout=float((ok.outcome == "timeout").mean()),
                viol_deck=float(ok.viol_deck.mean()), viol_over_g=float(ok.viol_over_g.mean()),
                first_error=(df["error"].dropna().iloc[0] if err else ""))


def clone(name: str, opps: list, check_opp: str, opts: dict, episodes: int, epochs: int, workers: int,
          outdir: str) -> dict:
    from ..rl.pretrain_bc import collect, fit, evaluate
    sixdof = bool(opts.get("sixdof"))
    tag = "6dof" if sixdof else "3dof"
    t0 = time.time()
    X, Y = collect(canon(name), tuple(opps), n_episodes=episodes, env_opts=opts, workers=workers)
    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, f"bc_{canon(name)}_{tag}")
    np.savez_compressed(base + "_data.npz", obs=X, act=Y, teacher=canon(name), sixdof=sixdof)
    pol = fit(X, Y, epochs=epochs, verbose=False)
    pol.save(base + ".npz")
    ev = evaluate(pol, check_opp, n_seeds=20, env_opts=opts)
    rep = dict(teacher=label(name), opponents=[label(o) for o in opps], env_opts=opts, sixdof=sixdof,
               episodes=episodes, samples=int(len(X)), r2=pol.bc_r2, eval={label(check_opp): ev},
               seconds=round(time.time() - t0))
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    return rep


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="기준선 BT 수용 검사 (C안 G1)")
    ap.add_argument("--bt", required=True, help="registry 이름 (btm, btmplus, school, tmpl, v2 …)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--quick", action="store_true", help="교전 시드 20, 복제 60교전·20회 반복 (형식 확인용)")
    ap.add_argument("--bc-epochs", type=int, default=1000,
                    help="행동 복제 반복 횟수. BT-M 처럼 스로틀이 계단인 BT 는 오래 배워야 한다(학습 상대 수집 3DOF R²: 200회 0.81, 1000회 0.88)")
    ap.add_argument("--no-6dof", action="store_true", help="3자유도만")
    ap.add_argument("--outdir", default="results/cplan")
    a = ap.parse_args()
    n_seeds, bc_eps, bc_epochs = (20, 60, 20) if a.quick else (100, 300, a.bc_epochs)
    name = canon(a.bt)
    plan = load_plan()
    train = [canon(o) for o in plan["train_opponents"]]
    partners = [o for o in train if o != name]              # ③ 의 "다른 학습 상대"
    opps = partners + [canon(o) for o in plan["heldout_opponents"] if canon(o) != name]
    print(f"[{label(name)}] 형식 점검", flush=True)
    iface, nodes = interface_checks(name)
    envs = [False] if a.no_6dof else [False, True]
    bat, bc, rules = {}, {}, {}
    for six in envs:
        tag = "6DOF" if six else "3DOF"
        rules[tag] = plan_env_opts(plan, "6dof" if six else "3dof")
        print(f"[{label(name)}] {tag} 교전 {2 * n_seeds * len(opps)}회 (규칙 {rules[tag]})", flush=True)
        bat[tag] = judge_battle(battle(name, opps, rules[tag], n_seeds, a.workers))
        print(f"[{label(name)}] {tag} 행동 복제 ({bc_eps}교전 수집)", flush=True)
        bc[tag] = clone(name, train, partners[0], rules[tag], bc_eps, bc_epochs, a.workers, os.path.join(a.outdir, "bc"))

    rows = [("형식", k, ok, d) for k, ok, d in iface]
    for tag in bat:
        b, c = bat[tag], bc[tag]
        rows += [(tag, f"① {b['n']}교전 오류 없음", b["errors"] == 0, f"오류 {b['errors']}회 " + b["first_error"][-200:]),
                 (tag, f"② 공중충돌 ≤ {TH['collision']:.0%}", b["collision"] <= TH["collision"], f"{b['collision']:.3f}"),
                 (tag, f"② 지면충돌(자기 추락) ≤ {TH['crash']:.0%}", b["crash"] <= TH["crash"], f"{b['crash']:.3f}"),
                 *[(tag, f"③ {label(p)} 상대 점수 {TH['partner_lo']}–{TH['partner_hi']}",
                    TH["partner_lo"] <= b["score"][label(p)] <= TH["partner_hi"], f"{b['score'][label(p)]:.3f}")
                   for p in partners],
                 (tag, f"④ 행동 복제 R² ≥ {TH['bc_r2']}", c["r2"]["weighted"] >= TH["bc_r2"],
                  f"{c['r2']['weighted']:.3f} (채널별 {[None if x is None else round(x, 3) for x in c['r2']['per_channel']]})")]
    passed = all(r[2] for r in rows)
    lines = [f"# 기준선 BT 수용 검사: {label(name)} ({time.strftime('%Y-%m-%d %H:%M')})", "",
             f"판정: **{'통과' if passed else '실패'}**" + (" (빠른 점검 — G1 판정에는 --quick 없이 다시 돌린다)" if a.quick else ""),
             "", "| 환경 | 항목 | 결과 | 값 |", "|---|---|---|---|"]
    lines += [f"| {e} | {k} | {'통과' if ok else '**실패**'} | {str(d).replace('|', '/')} |" for e, k, ok, d in rows]
    lines += ["", "## 참고 지표", "",
              f"| 환경 | {' | '.join(label(o) + ' 상대 점수' for o in opps)} | 접촉률 | WEZ 체류 [s] | 격추승 비율 | 시간종료 비율 | "
              f"하드덱 위반율 | 과G 위반율 | 복제본 vs {label(partners[0])} 점수 |",
              "|---|" + "---|" * (len(opps) + 7)]
    for tag in bat:
        b, c = bat[tag], bc[tag]
        lines.append(f"| {tag} | {' | '.join(f'{b_s:.3f}' for b_s in (b['score'][label(o)] for o in opps))} | "
                     f"{b['contact']:.3f} | {b['wez']:.2f} | {b['kill_win']:.3f} | {b['timeout']:.3f} | {b['viol_deck']:.4f} | "
                     f"{b['viol_over_g']:.4f} | {c['eval'][label(partners[0])]['score']:.3f} |")
    lines += ["", f"3자유도 시험 교전에서 발동한 노드: {', '.join(sorted(nodes))}",
              f"감독형이 가로챌 안전 노드: {', '.join(safety_nodes(make_bt(name)))}", "",
              f"환경 규칙: {rules}. 행동 복제 수집 상대: {', '.join(label(o) for o in train)}.",
              "기준(C안_실행명세서 4.1절 H4): ② ④ 의 수치는 제안 기준이다. ③ 은 2026-11-05 손형래 제안으로 BT-v2 상대 ≥ 0.45 에서 바꿨다. "
              "교전은 평가와 같은 방식(시드 10000부터, 진영 교대). 접촉률 = 어느 쪽이든 피해가 난 교전의 비율."]
    os.makedirs(a.outdir, exist_ok=True)
    md = os.path.join(a.outdir, f"accept_{name}.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(a.outdir, f"accept_{name}.json"), "w", encoding="utf-8") as f:
        json.dump(dict(bt=label(name), passed=passed, quick=a.quick, rows=[list(r) for r in rows],
                       battle=bat, bc=bc), f, indent=2, ensure_ascii=False, default=str)
    print("\n".join(lines))
    print(f"\n저장: {md}")


if __name__ == "__main__":
    main()
