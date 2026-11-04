"""
C안 E1·E2 수용 시험: 코드를 고친 뒤에도 기존 결과가 그대로 나오는가, 이어하기가 연속 실행과 같은가.

  train  : 학습기 4종(초기화형 train_es, 잔차형·게이팅형 train_es_hybrid, 파라미터 최적화형 train_es_bt)을
           본실험과 같은 인자로 2세대 돌려 저장된 history(results/es*, 시드 0)와 소수점 끝까지 비교
  eval   : 본실험 조건 일부를 같은 시드로 다시 평가해 results/main·main_hyb 의 교전 기록과 비교
  resume : 3자유도 20세대 연속 대 10세대 + 이어서 10세대 (θ·history 비교).  --sixdof-resume 은 6자유도 축소판

    python -m dfxai.cplan.repro_check --workers 10            (약 15분)
    python -m dfxai.cplan.repro_check --only eval --workers 10
"""
from __future__ import annotations
import argparse, json, os, shutil, sys, tempfile
import numpy as np
import pandas as pd

KEYS = ("fit_mean", "fit_max", "fit_theta", "accepted", "lr_eff")


def _same_hist(new: list, old: list, n: int) -> bool:
    return len(new) >= n and all(all(a[k] == b[k] for k in KEYS) for a, b in zip(new[:n], old[:n]))


def check_train(workers: int, tmp: str) -> list[tuple]:
    from ..rl import train_es, train_es_hybrid, train_es_bt
    out = []
    old = lambda p: json.load(open(p, encoding="utf-8"))
    _, h = train_es.train(generations=300, pop=32, episodes=8, sigma=0.05, lr=0.01, seed=0, workers=workers,
                          outdir=os.path.join(tmp, "es"), tag="seed0", init="results/es/bc_init.npz",
                          checkpoint_every=50, stop_after=2)
    out.append(("초기화형 train_es 시드 0", _same_hist(h, old("results/es/history_seed0.json"), 2)))
    _, h = train_es_hybrid.train("residual", generations=200, pop=32, sigma=0.05, lr=0.01, episodes=8, seed=0,
                                 workers=workers, outdir=os.path.join(tmp, "res"), tag="seed0", beta=0.35,
                                 stop_after=2)
    out.append(("잔차형 train_es_hybrid 시드 0", _same_hist(h, old("results/es_res/history_seed0.json"), 2)))
    _, h = train_es_hybrid.train("gating", generations=200, pop=32, sigma=0.05, lr=0.01, episodes=8, seed=0,
                                 workers=workers, outdir=os.path.join(tmp, "gate"), tag="seed0",
                                 rl_ckpt="results/es/ckpt_seed11_gen00300.npz", stop_after=2)
    out.append(("게이팅형 train_es_hybrid 시드 0", _same_hist(h, old("results/es_gate/history_seed0.json"), 2)))
    _, h = train_es_bt.train(generations=300, pop=32, sigma=0.10, lr=0.05, episodes=8, seed=0, workers=workers,
                             outdir=os.path.join(tmp, "bto"), tag="seed0", stop_after=2)
    out.append(("파라미터 최적화형 train_es_bt 시드 0", _same_hist(h, old("results/es_bto/history_seed0.json"), 2)))
    return out


def check_eval(workers: int, tmp: str, n_seeds: int = 5) -> list[tuple]:
    from ..experiments.run_eval import run
    names = ["BT-v1", "BT-v2", "BT-v3", "RL-s0-b300-a1.00", "HYB-s11-b300-a0.50", "SHD-s11-b300",
             "BTO-s0-b300", "RES-s0-b200", "GATE-s1-b200"]
    conds = []
    for d in ("results/main", "results/main_hyb"):
        with open(os.path.join(d, "manifest.json"), encoding="utf-8") as f:
            conds += [c for c in json.load(f)["conditions"] if c["name"] in names]
    conds = list({c["name"]: c for c in conds}.values())
    new = run(conds, (2, 3), n_seeds, workers, os.path.join(tmp, "eval"), record_episodes=0)
    ref = pd.concat([pd.read_csv("results/main/episodes.csv"), pd.read_csv("results/main_hyb/episodes.csv")])
    key = ["cond", "opponent", "seed", "side"]
    ref = ref[ref.cond.isin(names) & (ref.seed < 10_000 + n_seeds)].drop_duplicates(key)
    cols = ["score", "outcome", "duration", "damage_dealt", "damage_taken", "wez_time",
            "viol_deck", "viol_over_g", "viol_sep"]
    m = new.merge(ref, on=key, suffixes=("", "_ref"))
    ok = np.ones(len(m), bool)
    for c in cols:
        a, b = m[c], m[c + "_ref"]
        ok &= (a == b) if a.dtype == object else np.isclose(a, b, rtol=0, atol=1e-9)
    return [(f"본실험 교전 재현 ({len(names)}개 조건 × 상대 2 × 시드 {n_seeds} × 진영 2)",
             bool(ok.all() and len(m) == len(new)), f"{int(ok.sum())}/{len(new)}")]


def check_resume(workers: int, tmp: str, sixdof: bool = False) -> list[tuple]:
    from ..rl import train_es
    from ..rl.es_runtime import env_opts
    kw = dict(pop=32, episodes=8, sigma=0.05, lr=0.01, seed=3, workers=workers, init="results/es/bc_init.npz",
              tag="seed3", checkpoint_every=10)
    n = 20
    if sixdof:                       # 6자유도는 축소판: 개체군 8, 후보당 2교전, 4세대 = 2 + 2
        kw.update(pop=8, episodes=2, env_opts=env_opts(sixdof=True), checkpoint_every=2)
        n = 4
    a_dir, b_dir = os.path.join(tmp, f"r_full{sixdof}"), os.path.join(tmp, f"r_split{sixdof}")
    th_a, h_a = train_es.train(generations=n, outdir=a_dir, **kw)
    train_es.train(generations=n, outdir=b_dir, stop_after=n // 2, **kw)
    th_b, h_b = train_es.train(generations=n, outdir=b_dir, resume=True, **kw)
    same = bool(np.array_equal(th_a, th_b)) and _same_hist(h_b, h_a, n)
    c_a = np.load(os.path.join(a_dir, f"ckpt_seed3_gen{n:05d}.npz"))["flat"]
    c_b = np.load(os.path.join(b_dir, f"ckpt_seed3_gen{n:05d}.npz"))["flat"]
    lab = "6자유도 축소판 4세대 = 2 + 2" if sixdof else "3자유도 20세대 연속 = 10 + 10"
    return [(f"이어하기 ({lab})", same and bool(np.array_equal(c_a, c_b)))]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="C안 E1·E2 수용 시험")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", choices=("train", "eval", "resume"), default=None)
    ap.add_argument("--sixdof-resume", action="store_true", help="6자유도 이어하기 축소판도 시험")
    ap.add_argument("--out", default="results/cplan/checks/repro_check.json")
    a = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix="cplan_repro_")
    res = []
    try:
        if a.only in (None, "train"):
            res += check_train(a.workers, tmp)
        if a.only in (None, "eval"):
            res += check_eval(a.workers, tmp)
        if a.only in (None, "resume"):
            res += check_resume(a.workers, tmp)
            if a.sixdof_resume:
                res += check_resume(a.workers, tmp, sixdof=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n=== 결과 ===")
    for r in res:
        print(("통과 " if r[1] else "실패 ") + r[0] + (f"  [{r[2]}]" if len(r) > 2 else ""))
    from ..rl.es_runtime import git_info
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(dict(git=git_info(), results=[dict(test=r[0], passed=r[1], detail=r[2] if len(r) > 2 else "")
                                                for r in res]), f, indent=2, ensure_ascii=False)
    sys.exit(0 if all(r[1] for r in res) else 1)


if __name__ == "__main__":
    main()
