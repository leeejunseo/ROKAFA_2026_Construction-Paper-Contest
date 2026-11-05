"""
C안 분산 학습: 팀원 PC 에서 쓰는 명령 (E6). 안내서: paper/C안_분산학습_안내.md

    python -m dfxai.cplan.job selftest --env 6dof                     결정성 확인 (2세대, 지문 출력)
    python -m dfxai.cplan.job queue --csv 배정표.csv --pc PC3 --name 손형래 --upload-dir "G:/내 드라이브/C안_학습결과"
                                                                     배정표에서 이 PC 몫을 시드 순서대로 전부 실행
    python -m dfxai.cplan.job run --kind init --seed 3 --env 6dof --name 손형래      작업 하나만 실행
    python -m dfxai.cplan.job unpack 이준서_init_seed3.zip            (게이팅형) 다른 PC 의 초기화형 결과 받기

공통 규칙
  * 꺼졌다 켜지면 같은 명령을 다시 치면 된다. 끝난 작업은 건너뛰고, 하다 만 작업은 이어서 돈다(--resume).
  * 작업 하나가 끝나면 upload/<이름>_<종류>_seed<N>.zip 을 만들고, --upload-dir 를 주면 그곳에도 복사한다.
  * 학습 기간에는 git pull 을 하지 않는다. 시작할 때 태그(예: c-plan-v1)를 checkout 한 상태여야 한다.
"""
from __future__ import annotations
import os

# 결정성: numpy 를 불러오기 전에 정한다. 모든 PC 가 같은 BLAS 커널·같은 SIMD 경로를 쓰게 한다
# (AVX-512 가 있는 PC 와 없는 PC 의 부동소수점 끝자리가 달라지는 것을 막기 위함).
for _k, _v in (("OMP_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1"),
               ("OPENBLAS_CORETYPE", "Haswell"), ("NPY_DISABLE_CPU_FEATURES", "X86_V4"),
               ("MKL_CBWR", "COMPATIBLE"), ("PYTHONIOENCODING", "utf-8")):
    os.environ.setdefault(_k, _v)

import argparse, csv, glob, hashlib, json, re, shutil, subprocess, sys, tempfile, time, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PLAN_PATH = os.path.join(ROOT, "dfxai", "cplan", "plan.json")
ENV_PREFIX = {"6dof": "es6", "3dof": "es3"}


def load_plan(path: str | None = None) -> dict:
    """plan.json. 환경변수 CPLAN_PLAN 으로 다른 설정 파일을 줄 수 있다(배선 시험용 축소 설정)."""
    with open(path or os.environ.get("CPLAN_PLAN") or PLAN_PATH, encoding="utf-8") as f:
        return json.load(f)


def bc_init_path(plan: dict, env: str) -> str:
    return os.path.join("results", "cplan", "bc", f"bc_{plan['baseline']}_{env}.npz")


def outdir(plan: dict, kind: str, env: str) -> str:
    return os.path.join("results", f"{ENV_PREFIX[env]}_{plan['kinds'][kind]['outdir']}")


def command(plan: dict, kind: str, seed: int, env: str, workers: int, generations: int | None = None,
            pop: int | None = None, episodes: int | None = None, out: str | None = None,
            extra: list | None = None) -> list[str]:
    """작업 하나의 학습 명령. 모든 인자는 plan.json 에서 온다(팀원이 숫자를 손으로 치지 않게)."""
    k = plan["kinds"][kind]
    gens = generations or plan["generations"]
    cmd = [sys.executable, "-m", k["script"]]
    if kind in ("residual", "gating"):
        cmd += ["--kind", kind]
    cmd += ["--generations", str(gens), "--pop", str(pop or plan["pop"]),
            "--episodes", str(episodes or plan["episodes"]), "--sigma", str(k["sigma"]), "--lr", str(k["lr"]),
            "--seed", str(seed), "--tag", f"seed{seed}", "--workers", str(workers),
            "--checkpoint-every", str(plan["checkpoint_every"]), "--outdir", out or outdir(plan, kind, env),
            "--bt", plan["baseline"], "--opponents", *plan["train_opponents"], "--resume"]
    if kind == "init":
        cmd += ["--init", bc_init_path(plan, env)]
    if kind == "residual":
        cmd += ["--beta", str(k["beta"])]
    if kind == "gating":
        cmd += ["--rl-ckpt", os.path.join(outdir(plan, "init", env), f"ckpt_seed{seed}_gen{plan['generations']:05d}.npz")]
    if env == "6dof":
        cmd += ["--sixdof"]
        if plan.get("gun_cone_6dof"):
            cmd += ["--cone", *map(str, plan["gun_cone_6dof"])]
    elif plan.get("roll_rate_max_3dof"):
        cmd += ["--roll-rate-max", str(plan["roll_rate_max_3dof"])]
    if plan.get("accept_fresh"):
        cmd += ["--accept-fresh", "--accept-episodes", str(plan.get("accept_episodes") or plan["episodes"])]
    return cmd + (extra or [])


def _history(path: str) -> list:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


def is_done(plan: dict, kind: str, seed: int, env: str) -> bool:
    h = _history(os.path.join(outdir(plan, kind, env), f"history_seed{seed}.json"))
    return bool(h) and h[-1]["gen"] >= plan["generations"] and h[-1].get("final", False)


def zip_name(name: str, kind: str, seed: int, env: str) -> str:
    sfx = "" if env == "6dof" else "_3dof"
    return f"{name}_{kind}_seed{seed}{sfx}.zip"


def pack(plan: dict, name: str, kind: str, seed: int, env: str, upload_dir: str | None) -> str:
    """결과 폴더에서 이 작업의 파일(체크포인트·history·meta·로그)만 압축한다."""
    d = outdir(plan, kind, env)
    files = (sorted(glob.glob(os.path.join(d, f"ckpt_seed{seed}_gen*.npz")))
             + [os.path.join(d, f"{p}_seed{seed}{e}") for p, e in (("history", ".json"), ("meta", ".json"),
                                                                   ("log", ".txt"), ("params", ".json"))])
    os.makedirs("upload", exist_ok=True)
    zp = os.path.join("upload", zip_name(name, kind, seed, env))
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            if os.path.exists(f):
                z.write(f, arcname=os.path.join(os.path.basename(d), os.path.basename(f)))
    if upload_dir:
        os.makedirs(upload_dir, exist_ok=True)
        shutil.copy(zp, os.path.join(upload_dir, os.path.basename(zp)))
        print(f"업로드 폴더에 복사: {os.path.join(upload_dir, os.path.basename(zp))}", flush=True)
    return zp


def git_check(plan_tag: str | None) -> None:
    def g(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    head, desc = g("rev-parse", "HEAD"), g("describe", "--tags", "--always", "--dirty")
    dirty = g("status", "--porcelain", "--untracked-files=no", "--", "dfxai")
    print(f"코드: {desc} ({head[:8]})", flush=True)
    if dirty:
        print("[경고] dfxai/ 코드에 커밋하지 않은 수정이 있습니다. 배포 태그 그대로 돌려야 결과를 모을 수 있습니다.", flush=True)
    if plan_tag and not desc.startswith(plan_tag):
        print(f"[경고] 지금 코드가 태그 {plan_tag} 가 아닙니다 (git checkout {plan_tag}).", flush=True)


def run_job(plan: dict, kind: str, seed: int, env: str, name: str, workers: int,
            upload_dir: str | None = None) -> bool:
    if is_done(plan, kind, seed, env):
        zp = os.path.join("upload", zip_name(name, kind, seed, env))
        if not os.path.exists(zp):
            pack(plan, name, kind, seed, env, upload_dir)
        print(f"[건너뜀] {kind} 시드 {seed} ({env}) 는 이미 끝남", flush=True)
        return True
    if plan["kinds"][kind].get("needs") == "init":
        need = os.path.join(outdir(plan, "init", env), f"ckpt_seed{seed}_gen{plan['generations']:05d}.npz")
        if not os.path.exists(need):
            print(f"[대기] 게이팅형 시드 {seed} 는 초기화형 시드 {seed} 결과가 필요합니다: {need}\n"
                  f"       초기화형을 다른 PC 가 돌렸으면 그 zip 을 받아 'python -m dfxai.cplan.job unpack <zip>'", flush=True)
            return False
    if kind == "init" and not os.path.exists(bc_init_path(plan, env)):
        print(f"[오류] 행동 복제 초기값이 없습니다: {bc_init_path(plan, env)} (배포 태그에 들어 있어야 함)", flush=True)
        return False
    d = outdir(plan, kind, env)
    os.makedirs(d, exist_ok=True)
    cmd = command(plan, kind, seed, env, workers)
    log = os.path.join(d, f"log_seed{seed}.txt")
    print(f"[시작] {kind} 시드 {seed} ({env}), 작업자 {workers}  로그: {log}", flush=True)
    with open(log, "a", encoding="utf-8") as lf:
        lf.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd[1:])}\n")
        lf.flush()
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace", env=os.environ.copy())
        for line in p.stdout:
            if "JSBSim" in line or not line.strip():
                continue
            sys.stdout.write(line); sys.stdout.flush()
            lf.write(line); lf.flush()
        rc = p.wait()
    if rc != 0 or not is_done(plan, kind, seed, env):
        print(f"[중단] {kind} 시드 {seed}: 종료 코드 {rc}. 같은 명령을 다시 치면 이어서 돕니다.", flush=True)
        return False
    zp = pack(plan, name, kind, seed, env, upload_dir)
    print(f"[완료] {kind} 시드 {seed} → {zp}  (공유 드라이브 C안_학습결과/ 에 올리고 배정표를 '완료'로)", flush=True)
    return True


def read_assignments(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- 결정성 확인
def fingerprint(hist: list) -> str:
    vals = [(h["gen"], repr(h["fit_mean"]), repr(h["fit_max"]), repr(h["fit_theta"]), h["accepted"]) for h in hist]
    return hashlib.sha256(repr(vals).encode()).hexdigest()[:12]


def selftest(plan: dict, env: str, workers: int, kind: str = "init") -> str:
    """배포 설정 그대로 2세대를 임시 폴더에서 돌려 지문을 낸다. 전원 같아야 한다."""
    tmp = tempfile.mkdtemp(prefix="cplan_selftest_")
    try:
        cmd = command(plan, kind, 0, env, workers, generations=2, out=tmp)
        cmd = [c for c in cmd if c != "--resume"]
        t0 = time.time()
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           env=os.environ.copy())
        h = _history(os.path.join(tmp, "history_seed0.json"))
        if r.returncode != 0 or len(h) != 2:
            print(r.stdout[-3000:], r.stderr[-3000:])
            raise SystemExit("[실패] 시험 학습이 끝나지 않았습니다")
        fp = fingerprint(h)
        sec = time.time() - t0
        print(f"\n결정성 확인 ({env}, {kind} 시드 0, 2세대, {sec:.0f}초)")
        for x in h:
            print(f"  세대 {x['gen']}: fit_mean {x['fit_mean']!r}  fit_theta {x['fit_theta']!r}")
        print(f"지문: {fp}   ← 배정표의 '지문' 칸에 적으세요. 전원 같아야 합니다.")
        print(f"시험 시간: {sec:.0f}초   ← 이준서에게 알려 주세요. 이 PC 에서 {plan['generations']}세대 작업 하나는 "
              f"약 {sec * plan['generations'] / 2 / 3600:.1f}시간입니다(작업자 {workers}).")
        return fp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def unpack(zp: str) -> None:
    with zipfile.ZipFile(zp) as z:
        for m in z.namelist():
            if not re.match(r"^es[36]_\w+/[\w.\-]+$", m):
                raise SystemExit(f"예상하지 못한 경로가 들어 있습니다: {m}")
        z.extractall("results")
    print(f"풀었음: {zp} → results/")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(ROOT)
    ap = argparse.ArgumentParser(description="C안 분산 학습 작업")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("selftest", help="결정성 확인 (2세대)")
    s.add_argument("--env", choices=("6dof", "3dof"), default="6dof")
    s.add_argument("--kind", default="init")
    s.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    r = sub.add_parser("run", help="작업 하나 실행")
    r.add_argument("--kind", required=True, choices=("init", "residual", "gating", "bto"))
    r.add_argument("--seed", type=int, required=True)
    q = sub.add_parser("queue", help="배정표에서 이 PC 몫을 순서대로 실행")
    q.add_argument("--csv", required=True)
    q.add_argument("--pc", required=True)
    for p in (r, q):
        p.add_argument("--env", choices=("6dof", "3dof"), default="6dof")
        p.add_argument("--name", required=True, help="본인 이름 (압축 파일 이름에 들어감)")
        p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                       help="병렬 작업자 수. 기본 = 코어 수 − 1")
        p.add_argument("--upload-dir", default=None, help="완료 zip 을 복사할 폴더 (구글 드라이브 동기화 폴더 등)")
    u = sub.add_parser("unpack", help="다른 PC 의 결과 zip 을 results/ 에 풀기")
    u.add_argument("zip")
    st = sub.add_parser("status", help="이 PC 에 있는 학습 결과 현황")
    st.add_argument("--env", choices=("6dof", "3dof"), default="6dof")
    a = ap.parse_args()
    plan = load_plan()

    if a.cmd == "selftest":
        git_check(plan.get("tag"))
        selftest(plan, a.env, a.workers, a.kind)
    elif a.cmd == "run":
        git_check(plan.get("tag"))
        ok = run_job(plan, a.kind, a.seed, a.env, a.name, a.workers, a.upload_dir)
        sys.exit(0 if ok else 1)
    elif a.cmd == "queue":
        git_check(plan.get("tag"))
        rows = [x for x in read_assignments(a.csv) if x["pc"] == a.pc and x.get("env", a.env) == a.env]
        rows.sort(key=lambda x: int(x["order"]))
        print(f"{a.pc} 몫 {len(rows)}개: " + ", ".join(f"{x['kind']}-s{x['seed']}" for x in rows), flush=True)
        pending = []
        for x in rows:
            if not run_job(plan, x["kind"], int(x["seed"]), a.env, a.name, a.workers, a.upload_dir):
                pending.append(f"{x['kind']}-s{x['seed']}")
        print("남은 작업: " + (", ".join(pending) if pending else "없음 (이 PC 몫 전부 완료)"))
    elif a.cmd == "unpack":
        unpack(a.zip)
    elif a.cmd == "status":
        for kind in plan["kinds"]:
            d = outdir(plan, kind, a.env)
            for sp in sorted(glob.glob(os.path.join(d, "state_seed*.json"))):
                seed = int(re.search(r"seed(\d+)", sp).group(1))
                try:
                    with open(sp, encoding="utf-8") as f:
                        g = json.load(f)["gen"]
                except (OSError, ValueError, KeyError):
                    g = "?"
                print(f"{kind:9s} 시드 {seed:>2d}: {g}/{plan['generations']}세대"
                      + (" 완료" if is_done(plan, kind, seed, a.env) else ""))


if __name__ == "__main__":
    main()
