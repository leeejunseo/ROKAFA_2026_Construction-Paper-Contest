"""
교전 1회를 다시 돌려 Tacview 로 볼 수 있는 .acmi 파일로 내보냅니다.

    python -m dfxai.viz.replay --blue bt:2 --red bt:3 --seed 0
    python -m dfxai.viz.replay --blue hyb:results/es/ckpt_seed0_gen00400.npz:0.5 \
                               --red bt:3 --seeds 0 1 2 --outdir results/replay

정책 명세 형식
--------------
    bt:<버전>                 예) bt:2
    rl:<체크포인트>           예) rl:results/es/ckpt_seed0_gen00400.npz
    hyb:<체크포인트>:<alpha>  예) hyb:results/es/ckpt.npz:0.5

시드는 본실험과 동일한 난수원을 쓰므로, episodes.csv 에서 이상한 교전을
발견하면 그 seed 를 그대로 넣어 눈으로 확인할 수 있습니다.
"""
from __future__ import annotations
import argparse, os

import numpy as np

from ..env import DogfightEnv, run_episode
from ..agents.bt import BTPolicy
from ..agents.hybrid import MLPPolicy, HybridPolicy
from .acmi import write_acmi


def parse_spec(spec: str, bt_version: int = 2):
    """정책 명세 문자열 -> (정책, alpha, 표시이름)."""
    parts = spec.split(":")
    kind = parts[0].lower()
    if kind == "bt":
        v = int(parts[1]) if len(parts) > 1 else bt_version
        return BTPolicy(version=v), 0.0, f"BT-v{v}"
    if kind == "rl":
        return MLPPolicy.load(parts[1]), 1.0, "RL(alpha=1.00)"
    if kind in ("hyb", "hybrid"):
        alpha = float(parts[2]) if len(parts) > 2 else 0.5
        pol = HybridPolicy(BTPolicy(version=bt_version),
                           MLPPolicy.load(parts[1]), alpha=alpha)
        return pol, alpha, f"HYB-a{alpha:.2f}(BT-v{bt_version})"
    raise ValueError(f"알 수 없는 정책 명세: {spec}")


def export(blue_spec: str, red_spec: str, seed: int = 0,
           outdir: str = "results/replay", bt_version: int = 2,
           stride: int = 2, prefix: str = "") -> tuple[str, object]:
    """교전 1회를 실행하고 .acmi 로 저장. (경로, 결과) 를 돌려줍니다."""
    blue, a_b, nb = parse_spec(blue_spec, bt_version)
    red, a_r, nr = parse_spec(red_spec, bt_version)
    env = DogfightEnv(record_traj=True, traj_stride=stride)
    res = run_episode(blue, red, seed=seed, alpha=a_b, alpha_red=a_r, env=env)

    who = {1: nb, -1: nr, 0: "무승부"}[res.winner]
    brief = (f"{nb} (청) vs {nr} (홍) | seed={seed} | "
             f"{res.outcome} | 승자: {who} | "
             f"HP {res.hp_blue:.0f}:{res.hp_red:.0f} | "
             f"WEZ {res.wez_time_blue:.1f}s:{res.wez_time_red:.1f}s")
    safe = f"{prefix}{_slug(nb)}_vs_{_slug(nr)}_seed{seed}"
    path = os.path.join(outdir, safe + ".acmi")
    write_acmi(path, np.asarray(env.traj), env.events,
               blue_name=nb, red_name=nr,
               title=f"{nb} vs {nr} (seed {seed})", briefing=brief)
    return path, res


def _slug(s: str) -> str:
    keep = [c if (c.isalnum() or c in "-_.") else "-" for c in s]
    return "".join(keep).strip("-")


def main():
    ap = argparse.ArgumentParser(description="교전 재생용 ACMI 내보내기")
    ap.add_argument("--blue", default="bt:2", help="청군 정책 명세")
    ap.add_argument("--red", default="bt:3", help="홍군 정책 명세")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", nargs="*", type=int, default=None,
                    help="여러 시드를 한 번에 내보냅니다")
    ap.add_argument("--outdir", default="results/replay")
    ap.add_argument("--bt-version", type=int, default=2,
                    help="하이브리드에 들어가는 BT 버전")
    ap.add_argument("--stride", type=int, default=2,
                    help="기록 간격(적분 스텝 수). 2 = 0.1초")
    a = ap.parse_args()

    seeds = a.seeds if a.seeds else [a.seed]
    for s in seeds:
        path, res = export(a.blue, a.red, s, a.outdir, a.bt_version, a.stride)
        size = os.path.getsize(path) / 1024.0
        who = {1: "청", -1: "홍", 0: "무"}[res.winner]
        print(f"{path}  ({size:.0f} KB)  {res.outcome}  승자={who}  "
              f"{res.duration:.1f}s  HP {res.hp_blue:.0f}:{res.hp_red:.0f}")


if __name__ == "__main__":
    main()
