"""
C안 분산 학습: 엔지니어(이준서)가 쓰는 명령 (E6). 안내서: paper/C안_분산학습_안내.md

    python -m dfxai.cplan.admin assign --pcs PC1:이준서:11 PC2:손형래:3 PC3:박윤준:3 ... --out 배정표.csv
        작업 배정표(CSV) 생성. 시드 0–4 의 4종 → 5–9 → … 순서로 돌고, 게이팅형 시드 N 은 초기화형 시드 N 을
        돌린 PC 의 다음 바퀴에 들어간다(D7). 예상 시간은 PC 마다 selftest 시간(넷째 칸, 예: PC2:손형래:3:240)으로
        잡고, 없으면 6DOF 학습 1회 27코어시간(명세서 2절)으로 어림한다.
    python -m dfxai.cplan.admin collect --from "G:/내 드라이브/C안_학습결과" --env 6dof
        드라이브의 zip 을 results/es6_<종류>/ 에 풀고 검사한다: 300세대 완료, 커밋 해시 = 배포 태그,
        200세대 체크포인트 존재, 설정(기준선·상대·환경)이 plan.json 과 같음. → results/cplan/collect_6dof.md
"""
from __future__ import annotations
import argparse, csv, glob, json, os, re, subprocess, sys, zipfile
import pandas as pd

from .job import ROOT, load_plan, outdir, unpack, ENV_PREFIX

CORE_HOURS = {"6dof": 27.0, "3dof": 5.0}       # 학습 1회 (C안_실행명세서 2절, 3자유도 본실험 실측 30분 × 10코어)
KIND_ORDER = ("init", "residual", "bto")


def assign(pcs: list[str], env: str, seeds: int, block: int = 5, gens: int = 300) -> pd.DataFrame:
    pc = []
    for s in pcs:
        f = s.split(":")
        pid, owner, w = f[:3]
        # 넷째 칸 = 그 PC 의 selftest 시간[초]. 있으면 작업 1회 = 시험 시간 × 세대 수 / 2 (실측), 없으면 코어시간 어림
        hours = float(f[3]) * gens / 2 / 3600 if len(f) > 3 else CORE_HOURS[env] / int(w)
        pc.append(dict(pc=pid, owner=owner, workers=int(w), hours=hours, free=0.0))
    order = []
    blocks = [list(range(b, min(b + block, seeds))) for b in range(0, seeds, block)]
    for i, blk in enumerate(blocks):
        order += [(k, s) for s in blk for k in KIND_ORDER]
        if i > 0:
            order += [("gating", s) for s in blocks[i - 1]]
    order += [("gating", s) for s in blocks[-1]]
    where, rows = {}, []
    for n, (kind, seed) in enumerate(order, 1):
        if kind == "gating":
            p = where[("init", seed)]
        else:
            p = min(pc, key=lambda x: (x["free"] + x["hours"], x["hours"]))
        hours = p["hours"]
        start = p["free"]
        p["free"] += hours
        where[(kind, seed)] = p
        rows.append(dict(order=n, kind=kind, seed=seed, env=env, pc=p["pc"], owner=p["owner"], workers=p["workers"],
                         est_hours=round(hours, 1), est_start_h=round(start, 1), est_end_h=round(p["free"], 1),
                         status="대기", fingerprint="", note=""))
    return pd.DataFrame(rows)


def tag_commit(tag: str) -> str:
    return subprocess.run(["git", "rev-list", "-n", "1", tag], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


ZIP_RE = re.compile(r"^(?P<name>.+)_(?P<kind>init|residual|gating|bto)_seed(?P<seed>\d+)(?P<three>_3dof)?\.zip$")


def collect(src: str, env: str, plan: dict, tag: str | None) -> pd.DataFrame:
    want = tag_commit(tag) if tag else None
    if tag and not want:
        raise SystemExit(f"태그 {tag} 를 찾을 수 없습니다 (git fetch --tags)")
    rows = []
    for zp in sorted(glob.glob(os.path.join(src, "*.zip"))):
        m = ZIP_RE.match(os.path.basename(zp))
        if not m or (env == "3dof") != bool(m.group("three")):
            continue
        kind, seed = m.group("kind"), int(m.group("seed"))
        r = dict(kind=kind, seed=seed, by=m.group("name"), zip=os.path.basename(zp))
        try:
            unpack(zp)
        except (zipfile.BadZipFile, SystemExit) as e:
            rows.append({**r, "ok": False, "problem": f"압축 오류: {e}"}); continue
        d = outdir(plan, kind, env)
        prob = []
        try:
            with open(os.path.join(d, f"history_seed{seed}.json"), encoding="utf-8") as f:
                h = json.load(f)
            r["last_gen"] = h[-1]["gen"]
            r["fit_theta_last"] = h[-1]["fit_theta"]
            if h[-1]["gen"] < plan["generations"] or not h[-1].get("final"):
                prob.append(f"{h[-1]['gen']}세대에서 끝남")
        except (OSError, ValueError, IndexError):
            prob.append("history 없음")
        for g in plan["keep_checkpoints"]:
            if not os.path.exists(os.path.join(d, f"ckpt_seed{seed}_gen{g:05d}.npz")):
                prob.append(f"{g}세대 체크포인트 없음")
        try:
            with open(os.path.join(d, f"meta_seed{seed}.json"), encoding="utf-8") as f:
                meta = json.load(f)
            commit = meta["git"]["commit"]
            r["commit"] = commit[:8]
            if want and commit != want:
                prob.append(f"커밋 {commit[:8]} ≠ 태그 {want[:8]}")
            if meta["git"].get("code_dirty"):
                prob.append("코드 수정본으로 학습")
            for res in meta.get("resumes", []):
                if want and res["git"]["commit"] != want:
                    prob.append(f"이어하기 때 커밋이 다름 ({res['git']['commit'][:8]})")
            args = meta["args"]
            if str(args.get("bt")) != plan["baseline"]:
                prob.append(f"기준선 {args.get('bt')} ≠ {plan['baseline']}")
            if list(map(str, args.get("opponents", []))) != plan["train_opponents"]:
                prob.append(f"학습 상대 {args.get('opponents')} ≠ {plan['train_opponents']}")
            if bool(args.get("sixdof")) != (env == "6dof"):
                prob.append("환경(3DOF/6DOF)이 다름")
            if int(args.get("seed", -1)) != seed:
                prob.append(f"시드 {args.get('seed')} ≠ 파일 이름 {seed}")
            r["resumes"] = len(meta.get("resumes", [])) - 1 if meta.get("resumes") else 0
            r["numpy"], r["python"] = meta.get("numpy"), meta.get("python")
        except (OSError, ValueError, KeyError) as e:
            prob.append(f"meta 없음/손상 ({e})")
        rows.append({**r, "ok": not prob, "problem": "; ".join(prob)})
    return pd.DataFrame(rows)


def report(df: pd.DataFrame, plan: dict, env: str, path: str) -> str:
    from ..analysis.paper import md_table
    lines = [f"# C안 학습 결과 수집 ({env})", ""]
    if df.empty:
        lines.append("모인 zip 이 없습니다.")
    else:
        kinds = list(plan["kinds"])
        piv = df.pivot_table(index="seed", columns="kind", values="ok", aggfunc="max").reindex(columns=kinds)
        done = piv.fillna(False).astype(bool)
        full = [int(s) for s in done.index if done.loc[s].all()]
        prefix = 0
        while prefix in full:
            prefix += 1
        lines += [f"정상 {int(df.ok.sum())}개 / 받은 zip {len(df)}개.  네 종류가 모두 정상인 시드: {full}",
                  f"G4 동결 시 쓸 수 있는 시드(0부터 빠짐없이 이어지는 구간): 0–{prefix - 1}" if prefix else
                  "G4 동결 시 쓸 수 있는 시드: 아직 없음", "",
                  "## 종류별", "", md_table(df.groupby("kind").agg(받음=("ok", "size"), 정상=("ok", "sum")).reset_index()), "",
                  "## 문제 있는 결과", ""]
        bad = df[~df.ok]
        lines.append(md_table(bad[["kind", "seed", "by", "problem"]]) if len(bad) else "없음")
        lines += ["", "## 전체", "", md_table(df.drop(columns=["zip"]).sort_values(["kind", "seed"]))]
    text = "\n".join(lines) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(ROOT)
    ap = argparse.ArgumentParser(description="C안 분산 학습 관리")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("assign")
    s.add_argument("--pcs", nargs="+", required=True, help="PC아이디:담당자:작업자수[:selftest초]  (예: PC1:이준서:11:120)")
    s.add_argument("--env", choices=("6dof", "3dof"), default="6dof")
    s.add_argument("--seeds", type=int, default=None)
    s.add_argument("--out", default="results/cplan/배정표.csv")
    c = sub.add_parser("collect")
    c.add_argument("--from", dest="src", required=True, help="zip 이 모인 폴더 (드라이브 C안_학습결과/)")
    c.add_argument("--env", choices=("6dof", "3dof"), default="6dof")
    c.add_argument("--tag", default=None, help="배포 태그 (기본: plan.json 의 tag)")
    a = ap.parse_args()
    plan = load_plan()
    if a.cmd == "assign":
        df = assign(a.pcs, a.env, a.seeds or plan["seeds"], gens=plan["generations"])
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        df.to_csv(a.out, index=False, encoding="utf-8-sig")
        per = df.groupby(["pc", "owner"]).agg(작업=("kind", "size"), 예상시간=("est_hours", "sum"),
                                              끝나는시각_h=("est_end_h", "max")).reset_index()
        print(per.to_string(index=False))
        print(f"\n전체가 끝나는 데 약 {df.est_end_h.max():.0f}시간 ({df.est_end_h.max() / 24:.1f}일, 24시간 가동). 저장: {a.out}")
    else:
        df = collect(a.src, a.env, plan, a.tag or plan.get("tag"))
        os.makedirs("results/cplan", exist_ok=True)
        df.to_csv(f"results/cplan/collect_{a.env}.csv", index=False, encoding="utf-8-sig")
        print(report(df, plan, a.env, f"results/cplan/collect_{a.env}.md"))


if __name__ == "__main__":
    main()
