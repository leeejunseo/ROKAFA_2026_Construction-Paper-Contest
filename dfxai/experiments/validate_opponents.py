"""
교범 기반 상대 4종(BT-M, BT-E, BT-R, BT-M+) 상호 검증.

    python -m dfxai.experiments.validate_opponents --workers 4            # 3자유도 + 6자유도
    python -m dfxai.experiments.validate_opponents --workers 4 --skip-6dof

모든 짝(6쌍)을 환경마다 400교전(시드 10000~10099 × 진영 교대) 치른다.
판정 기준 (새 상대 BT-E, BT-R 에 적용, BT-M+ 는 참고)
    O1 오류 없음 / O2 공중충돌 ≤ 10% / O3 자기 추락 ≤ 5%
    O4 BT-M 상대 점수 0.35~0.65 (어느 한쪽이 압도하지 않음, 3·6자유도 각각)
참고: 하드덱·과G 위반율, 행동 특성(평균 등가속도, 평균 거리, 평균 하중), 노드 발동 빈도, 결과 유형.
"""
from __future__ import annotations
import argparse, collections, itertools, json, math, os, time, traceback
from multiprocessing import Pool

import numpy as np

from ..dynamics import air_density
from ..config import RHO0
from ..geometry import relative_geometry
from ..agents.bt_manual import ManualBTPolicy, BTMParams
from ..agents.bt_energy import EnergyBTPolicy
from ..agents.bt_radius import RadiusBTPolicy
from .run_eval import make_env

POLICIES = {
    "BT-M": lambda: ManualBTPolicy(BTMParams.core()),
    "BT-E": EnergyBTPolicy,
    "BT-R": RadiusBTPolicy,
    "BT-M+": lambda: ManualBTPolicy(BTMParams.full()),
}
ROLE = {"BT-M": "학습", "BT-E": "학습", "BT-R": "평가 전용", "BT-M+": "평가 전용"}
SEED0, N_SEEDS = 10_000, 100
NEW = ("BT-E", "BT-R")


def _job(args):
    a, b, seed, sixdof = args
    try:
        env = make_env({"sixdof": True} if sixdof else {})
        pa, pb = POLICIES[a](), POLICIES[b]()
        ob, orr = env.reset(seed=seed)
        stat = {a: collections.defaultdict(float), b: collections.defaultdict(float)}
        nodes = {a: collections.Counter(), b: collections.Counter()}
        n = 0
        for _ in range(env.max_steps + 1):
            ua, ub = pa.act(ob), pb.act(orr)
            nodes[a][pa.last_node] += 1; nodes[b][pb.last_node] += 1
            ob, orr, done = env.step(ua, ub)
            r = relative_geometry(env.blue, env.red)["r"]
            for name, st in ((a, env.blue), (b, env.red)):
                stat[name]["eas"] += st.v * math.sqrt(air_density(st.h) / RHO0)
                stat[name]["n"] += st.n
                stat[name]["h"] += st.h
            stat[a]["r"] += r; n += 1
            if done:
                break
        res = env.result()
        return dict(ok=True, a=a, b=b, seed=seed, sixdof=sixdof, winner=res.winner, outcome=res.outcome,
                    dmg=res.damage_dealt + res.damage_taken,
                    viol={a: res.violations_blue, b: res.violations_red},
                    beh={k: {m: v / max(n, 1) for m, v in d.items()} for k, d in stat.items()},
                    nodes={k: dict(v) for k, v in nodes.items()})
    except Exception:
        return dict(ok=False, a=a, b=b, seed=seed, sixdof=sixdof, error=traceback.format_exc()[-300:])


def run(sixdof: bool, workers: int) -> list:
    pairs = list(itertools.combinations(POLICIES, 2))
    # 진영 교대: (a 청, b 홍) 과 (b 청, a 홍) 을 같은 시드로
    jobs = [(a, b, SEED0 + i, sixdof) for x, y in pairs for i in range(N_SEEDS) for a, b in ((x, y), (y, x))]
    if workers > 1:
        with Pool(workers) as p:
            return p.map(_job, jobs, chunksize=4)
    return [_job(j) for j in jobs]


def summarize(rows: list) -> dict:
    out = {"pairs": {}, "policy": {}}
    for x, y in itertools.combinations(POLICIES, 2):
        rr = [r for r in rows if {r["a"], r["b"]} == {x, y}]
        ok = [r for r in rr if r["ok"]]
        sc = np.array([(1.0 if r["winner"] == 1 else .5 if r["winner"] == 0 else 0.0) if r["a"] == x else
                       (1.0 if r["winner"] == -1 else .5 if r["winner"] == 0 else 0.0) for r in ok])
        rng = np.random.default_rng(0)
        boot = [sc[rng.integers(0, len(sc), len(sc))].mean() for _ in range(2000)]
        crash = {p: np.mean([r["outcome"] == "crash" and ((r["winner"] == -1) == (r["a"] == p)) for r in ok]) for p in (x, y)}
        oc = collections.Counter(r["outcome"] for r in ok)
        out["pairs"][f"{x} vs {y}"] = dict(
            n=len(rr), errors=len(rr) - len(ok), score_x=float(sc.mean()),
            ci=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
            collision=oc["collision"] / len(ok), kill=oc["gun_kill"] / len(ok), timeout=oc["timeout"] / len(ok),
            contact=float(np.mean([r["dmg"] > 0 for r in ok])),
            crash={p: float(v) for p, v in crash.items()})
    for p in POLICIES:
        ok = [r for r in rows if r["ok"] and p in (r["a"], r["b"])]
        nodes = collections.Counter()
        for r in ok:
            nodes.update(r["nodes"][p])
        tot = sum(nodes.values()) or 1
        out["policy"][p] = dict(
            nodes_count=POLICIES[p]().complexity()["bt_nodes"],
            deck=float(np.mean([r["viol"][p]["deck"] for r in ok])),
            over_g=float(np.mean([r["viol"][p]["over_g"] for r in ok])),
            eas=float(np.mean([r["beh"][p]["eas"] for r in ok])),
            load=float(np.mean([r["beh"][p]["n"] for r in ok])),
            alt=float(np.mean([r["beh"][p]["h"] for r in ok])),
            node_freq={k: v / tot for k, v in nodes.most_common(6)})
    return out


def judge(s3: dict, s6: dict | None) -> list:
    v = []
    for tag, s in (("3자유도", s3), ("6자유도", s6)):
        if s is None:
            continue
        for p in NEW:
            pr = {k: d for k, d in s["pairs"].items() if p in k.split(" vs ")}
            err = sum(d["errors"] for d in pr.values())
            col = max(d["collision"] for d in pr.values())
            cr = max(d["crash"][p] for d in pr.values())
            key = f"BT-M vs {p}"
            sc_m = s["pairs"][key]["score_x"]
            sc_p = 1.0 - sc_m
            v += [(f"O1 {tag} {p} 오류 없음 (3쌍 × 400교전)", f"{err}회", err == 0),
                  (f"O2 {tag} {p} 공중충돌 ≤ 10% (짝별 최대)", f"{col:.3f}", col <= 0.10),
                  (f"O3 {tag} {p} 자기 추락 ≤ 5% (짝별 최대)", f"{cr:.3f}", cr <= 0.05),
                  (f"O4 {tag} {p} 의 BT-M 상대 점수 0.35~0.65", f"{sc_p:.3f}", 0.35 <= sc_p <= 0.65)]
    return v


def report(s3, s6, verdict, path):
    L = ["# 교범 기반 상대 4종 상호 검증 (자동 생성)", "",
         "평가: 짝마다 시드 10000–10099 × 진영 교대 = 400교전, 6쌍. 점수는 표의 왼쪽 모델 기준(승 1, 무 0.5).", "",
         "## 1. 판정 (새 상대 BT-E, BT-R)", "", "| 기준 | 값 | 판정 |", "|---|---|---|"]
    L += [f"| {a} | {b} | {'충족' if c else '미충족'} |" for a, b, c in verdict]
    for tag, s in (("3자유도", s3), ("6자유도", s6)):
        if s is None:
            continue
        L += ["", f"## 2. 짝별 결과 — {tag}", "",
              "| 짝 | 왼쪽 점수 [95% 구간] | 접촉률 | 격추 | 시간종료 | 공중충돌 | 추락(왼/오) |", "|---|---|---|---|---|---|---|"]
        for k, d in s["pairs"].items():
            x, y = k.split(" vs ")
            L.append(f"| {k} | {d['score_x']:.3f} [{d['ci'][0]:.3f}, {d['ci'][1]:.3f}] | {d['contact']:.2f} | {d['kill']:.2f} | "
                     f"{d['timeout']:.2f} | {d['collision']:.3f} | {d['crash'][x]:.3f} / {d['crash'][y]:.3f} |")
        L += ["", f"### 모델별 행동 특성 — {tag}", "",
              "| 모델 | 역할 | 노드 | 평균 등가속도 [m/s] | 평균 하중 [G] | 평균 고도 [m] | 하드덱 위반 | 과G 위반 | 주요 노드 |", "|---|---|---|---|---|---|---|---|---|"]
        for p, d in s["policy"].items():
            top = ", ".join(f"{k} {v:.2f}" for k, v in list(d["node_freq"].items())[:4])
            L.append(f"| {p} | {ROLE[p]} | {d['nodes_count']} | {d['eas']:.0f} | {d['load']:.2f} | {d['alt']:.0f} | "
                     f"{d['deck']:.4f} | {d['over_g']:.4f} | {top} |")
    txt = "\n".join(L) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)
    return txt


def main():
    ap = argparse.ArgumentParser(description="교범 기반 상대 4종 상호 검증")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--skip-6dof", action="store_true")
    ap.add_argument("--outdir", default="results/opponents")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    t0 = time.time()
    s3 = summarize(run(False, a.workers))
    print(f"3자유도 완료 ({time.time() - t0:.0f}s)", flush=True)
    s6 = None
    if not a.skip_6dof:
        s6 = summarize(run(True, a.workers))
        print(f"6자유도 완료 ({time.time() - t0:.0f}s)", flush=True)
    v = judge(s3, s6)
    with open(os.path.join(a.outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(dict(s3=s3, s6=s6, verdict=v), f, ensure_ascii=False, indent=1)
    txt = report(s3, s6, v, os.path.join(a.outdir, "report.md"))
    try:
        print(txt)
    except UnicodeEncodeError:                     # Windows cp949 콘솔
        print(f"보고서 저장: {os.path.join(a.outdir, 'report.md')}")


if __name__ == "__main__":
    main()
