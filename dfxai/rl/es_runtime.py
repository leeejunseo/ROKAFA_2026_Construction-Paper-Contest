"""
ES 학습기 세 개(train_es, train_es_hybrid, train_es_bt)가 함께 쓰는 실행 도구 (C안 E1·E2·E6).

  * 환경 옵션    : --sixdof 이면 6자유도(JSBSim F-16). 비면 3자유도이며 기존 결과를 그대로 재현한다.
  * 이어하기     : 매 세대 끝에 state_<tag>.json/.npz 를 원자적으로 저장한다. --resume 이면 거기서 잇는다.
                   세대 번호, 난수 상태, 실효 학습률(lr_eff), 수용 횟수, history 를 모두 복원하므로
                   "20세대 연속"과 "10세대 + 이어서 10세대"의 결과가 같다.
  * 실행 기록    : meta_<tag>.json 에 커밋 해시·태그·인자·라이브러리 버전·시작/끝 시각을 남긴다.
  * 새 시드 수용 검사(fresh_check): --accept-fresh 일 때 갱신 수용 여부를 기울기와 다른 교전으로 판정한다.
"""
from __future__ import annotations
import json, os, platform, subprocess, sys, time
import numpy as np


def env_opts(sixdof: bool = False, roll_rate_max: float | None = None) -> dict:
    """학습 환경 옵션. 3자유도 기본값은 빈 dict (run_eval.make_env 와 같은 형식)."""
    o = {}
    if roll_rate_max:
        o["roll_rate_max_deg"] = float(roll_rate_max)
    if sixdof:
        o["sixdof"] = True
    return o


def make_env(opts: dict | None):
    """빈 옵션이면 기존과 똑같은 DogfightEnv(). 6자유도는 작업자 프로세스마다 하나를 재사용."""
    if not opts:
        from ..env import DogfightEnv
        return DogfightEnv()
    from ..experiments.run_eval import make_env as _make
    return _make(opts)


def fresh_check(rng: np.random.Generator, opponents, n: int):
    """수용 검사용 새 시드·상대 (기울기 추정과 같은 방식으로 뽑되 다른 교전)."""
    seeds = rng.integers(0, 10 ** 6, n)
    opps = np.array([opponents[i % len(opponents)] for i in range(n)])
    rng.shuffle(opps)
    return seeds, opps


def pmap(pool, fn, jobs: list) -> list:
    return pool.map(fn, jobs) if pool else [fn(j) for j in jobs]


# ---------------------------------------------------------------- 이어하기
def _paths(outdir: str, tag: str):
    return (os.path.join(outdir, f"state_{tag}.json"), os.path.join(outdir, f"state_{tag}.npz"))


def save_state(outdir: str, tag: str, gen: int, theta: np.ndarray, rng: np.random.Generator,
               lr_eff: float, n_accept: int, history: list, extra: dict | None = None) -> None:
    """한 세대가 끝난 직후의 학습 상태. npz(θ) 를 먼저, json 을 나중에 바꿔 둘이 어긋나지 않게 한다."""
    pj, pn = _paths(outdir, tag)
    tmp = pn + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, theta=theta)
    os.replace(tmp, pn)
    st = dict(gen=int(gen), lr_eff=float(lr_eff), n_accept=int(n_accept),
              rng=rng.bit_generator.state, history=history, extra=extra or {},
              theta_sum=float(np.sum(theta)), saved=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(pj + ".tmp", "w", encoding="utf-8") as f:
        json.dump(st, f)
    os.replace(pj + ".tmp", pj)


def load_state(outdir: str, tag: str, rng: np.random.Generator):
    """저장된 상태가 있으면 (gen, θ, lr_eff, n_accept, history) 를 돌려주고 rng 를 되돌린다. 없으면 None."""
    pj, pn = _paths(outdir, tag)
    if not (os.path.exists(pj) and os.path.exists(pn)):
        return None
    with open(pj, encoding="utf-8") as f:
        st = json.load(f)
    theta = np.load(pn)["theta"]
    if abs(float(np.sum(theta)) - st["theta_sum"]) > 1e-9 * max(1.0, abs(st["theta_sum"])):
        raise RuntimeError(f"{pn} 와 {pj} 가 서로 다른 세대의 것입니다. 마지막 체크포인트에서 다시 시작하세요")
    rng.bit_generator.state = st["rng"]
    return st["gen"], theta, st["lr_eff"], st["n_accept"], st["history"]


# ---------------------------------------------------------------- 실행 기록
def git_info() -> dict:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def g(*a):
        try:
            return subprocess.run(["git", *a], cwd=root, capture_output=True, text=True,
                                  timeout=20).stdout.strip()
        except Exception:
            return ""
    dirty = g("status", "--porcelain", "--untracked-files=no", "--", "dfxai")
    return dict(commit=g("rev-parse", "HEAD"), describe=g("describe", "--tags", "--always", "--dirty"),
                code_dirty=bool(dirty))


def write_meta(outdir: str, tag: str, args: dict, finished: bool = False, gen: int | None = None) -> None:
    """meta_<tag>.json: 어떤 코드·설정으로 학습했는지. 이어하기 때마다 resumes 에 한 줄씩 덧붙인다."""
    path = os.path.join(outdir, f"meta_{tag}.json")
    old = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            old = json.load(f)
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    gi = git_info()
    if not old:
        import numpy
        try:
            import jsbsim
            jv = jsbsim.__version__
        except Exception:
            jv = None
        old = dict(tag=tag, args=args, git=gi, started=now, python=sys.version.split()[0],
                   numpy=numpy.__version__, jsbsim=jv, platform=platform.platform(),
                   cpu_count=os.cpu_count(), resumes=[],
                   env_vars={k: os.environ.get(k) for k in
                             ("OMP_NUM_THREADS", "OPENBLAS_CORETYPE", "NPY_DISABLE_CPU_FEATURES", "MKL_CBWR")})
    elif not finished:
        old["resumes"].append(dict(at=now, gen=gen, git=gi))
        if gi["commit"] != old["git"]["commit"]:
            print(f"[경고] 처음 학습한 커밋({old['git']['commit'][:8]})과 지금 커밋({gi['commit'][:8]})이 다릅니다",
                  flush=True)
    if finished:
        old["finished"] = now
        old["last_gen"] = gen
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(old, f, indent=2, ensure_ascii=False)
    os.replace(path + ".tmp", path)
