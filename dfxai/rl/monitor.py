"""
학습 진행 뷰어.

    python -m dfxai.rl.monitor --esdir results/es --watch 30
    -> results/es/monitor.html 을 브라우저로 열어 두면 30초마다 자동 갱신됩니다.

무엇을 보여주나
---------------
(a) 셰이핑 적합도 곡선 — 학습 로그(20세대마다)와 history_*.json(세대마다) 병합.
(b) 실제 전투 점수 곡선 — 체크포인트가 생길 때마다 BT-v2·BT-v3 상대로
    짧은 진영교대 평가(기본 20시드 x 2진영)를 돌려 점수(승1·무0.5·패0)를 찍습니다.
    한 번 평가한 체크포인트는 monitor_cache.json 에 저장해 다시 돌리지 않습니다.
    적합도는 셰이핑 항이 섞여 있어 오르는 것처럼 보여도 실제로는 못 이길 수
    있으므로, 논문에 쓸 '학습이 됐다'는 판단은 (b) 로 하십시오.
    BT-v1·v3 가 같은 상대에게 받는 점수를 기준선(점선)으로 함께 그립니다.

빠른 평가는 학습과 CPU 를 나눠 쓰므로 시드 수를 크게 늘리지 마십시오.
본실험(100시드)은 run_eval 로 따로 돌립니다.
"""
from __future__ import annotations
import argparse, base64, glob, io, json, os, re, time
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..env import run_episode
from ..agents.bt import BTPolicy
from ..agents.hybrid import MLPPolicy, HybridPolicy

_LOG = re.compile(r"\[gen\s+(\d+)\]\s+fit mean=\s*([-\d.]+)\s+max=\s*([-\d.]+)"
                  r"\s+alpha>=([\d.]+)\s+\((\d+)s\)")
_CK = re.compile(r"ckpt_(.+)_gen(\d+)\.npz$")


# ------------------------------------------------------------ 데이터 수집
def read_progress(esdir: str) -> dict[str, dict]:
    """태그별 {gen: [...], fit_mean: [...], fit_max: [...], elapsed: [...], done: bool}."""
    runs: dict[str, dict] = {}
    for log in sorted(glob.glob(os.path.join(esdir, "train_*.log"))):
        tag = os.path.basename(log)[len("train_"):-len(".log")]
        rows = []
        with open(log, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = _LOG.search(line)
                if m:
                    rows.append((int(m[1]), float(m[2]), float(m[3]), float(m[4]), float(m[5])))
        runs[tag] = dict(gen=[r[0] for r in rows], fit_mean=[r[1] for r in rows],
                         fit_max=[r[2] for r in rows], alpha_lo=[r[3] for r in rows],
                         elapsed=[r[4] for r in rows], done=False, source="log")
    for hist in sorted(glob.glob(os.path.join(esdir, "history_*.json"))):
        tag = os.path.basename(hist)[len("history_"):-len(".json")]
        try:
            h = json.load(open(hist, encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue                        # 쓰는 도중에 읽힌 경우
        if not h:
            continue
        runs[tag] = dict(gen=[r["gen"] for r in h], fit_mean=[r["fit_mean"] for r in h],
                         fit_max=[r["fit_max"] for r in h],
                         alpha_lo=[r.get("alpha_lo", np.nan) for r in h],
                         elapsed=[r["elapsed"] for r in h],
                         done=bool(h[-1].get("final", False)) or runs.get(tag, {}).get("done", False),
                         source="history")
    return runs


def list_checkpoints(esdir: str) -> list[tuple[str, int, str]]:
    out = []
    for p in glob.glob(os.path.join(esdir, "ckpt_*_gen*.npz")):
        m = _CK.search(os.path.basename(p))
        if m:
            out.append((m[1], int(m[2]), p))
    return sorted(out)


# ------------------------------------------------------------ 빠른 평가
def quick_score(policy, alpha: float, opponent: int, n_seeds: int, seed0: int = 50_000) -> float:
    """진영 교대 짝지은 점수. 시드는 본실험(10000~)과 겹치지 않게 둡니다."""
    sc = []
    for s in range(seed0, seed0 + n_seeds):
        r = run_episode(policy, BTPolicy(version=opponent), seed=s, alpha=alpha, alpha_red=0.0)
        sc.append(1.0 if r.winner == 1 else (0.5 if r.winner == 0 else 0.0))
        r = run_episode(BTPolicy(version=opponent), policy, seed=s, alpha=0.0, alpha_red=alpha)
        sc.append(1.0 if r.winner == -1 else (0.5 if r.winner == 0 else 0.0))
    return float(np.mean(sc))


def update_cache(esdir: str, n_seeds: int, opponents=(2, 3), alphas=(1.0, 0.5),
                 bt_version: int = 2, max_new: int = 3) -> dict:
    """새 체크포인트만 평가해 캐시에 추가. 한 번에 max_new 개까지(학습과 CPU 공유)."""
    path = os.path.join(esdir, "monitor_cache.json")
    cache = {}
    if os.path.exists(path):
        try:
            cache = json.load(open(path, encoding="utf-8"))
        except json.JSONDecodeError:
            cache = {}
    cache.setdefault("n_seeds", n_seeds)
    cache.setdefault("baseline", {})
    cache.setdefault("ckpt", {})

    for v in (1, 3):                                  # 기준선: BT-v1, BT-v3
        for ov in opponents:
            k = f"BT-v{v}|vs{ov}"
            if k not in cache["baseline"]:
                cache["baseline"][k] = quick_score(BTPolicy(version=v), 0.0, ov, n_seeds)

    n_done = 0
    for tag, gen, p in list_checkpoints(esdir):
        key = os.path.basename(p)
        if key in cache["ckpt"]:
            continue
        if n_done >= max_new:
            break
        rl = MLPPolicy.load(p)
        rec = dict(tag=tag, gen=gen)
        for a in alphas:
            pol = rl if a >= 1.0 else HybridPolicy(BTPolicy(version=bt_version), rl, alpha=a)
            for ov in opponents:
                rec[f"a{a:.2f}|vs{ov}"] = quick_score(pol, a, ov, n_seeds)
        cache["ckpt"][key] = rec
        n_done += 1
        with open(path, "w", encoding="utf-8") as f:   # 중간 저장 (중단돼도 보존)
            json.dump(cache, f, indent=1)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=1)
    return cache


# ------------------------------------------------------------ 그림
def render_figure(runs: dict, cache: dict, generations: int, alphas=(1.0, 0.5),
                  opponents=(2, 3)) -> bytes:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    tags = sorted(runs)
    cmap = plt.get_cmap("tab10")
    color = {t: cmap(i % 10) for i, t in enumerate(tags)}

    # (a) 적합도
    ax = axes[0]
    for t in tags:
        r = runs[t]
        if not r["gen"]:
            continue
        ax.plot(r["gen"], r["fit_mean"], color=color[t], lw=1.3, label=f"{t} mean")
        ax.plot(r["gen"], r["fit_max"], color=color[t], lw=0.8, ls="--", alpha=0.6)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlim(0, generations); ax.set_xlabel("generation"); ax.set_ylabel("shaped fitness")
    ax.set_title("(a) ES fitness (solid mean, dashed pop. max)", fontsize=9)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    # (b)(c) 실제 점수 — 상대별 패널
    ck = list(cache.get("ckpt", {}).values())
    for ax, ov in zip(axes[1:], opponents):
        for t in tags:
            rows = sorted([c for c in ck if c["tag"] == t], key=lambda c: c["gen"])
            for a, ls, mk in zip(alphas, ("-", ":"), ("o", "^")):
                k = f"a{a:.2f}|vs{ov}"
                xs = [c["gen"] for c in rows if k in c]
                ys = [c[k] for c in rows if k in c]
                if xs:
                    ax.plot(xs, ys, ls, marker=mk, ms=4, lw=1.2, color=color[t],
                            label=f"{t} α={a:.2f}")
        for v, c_ in ((1, "0.6"), (3, "0.3")):
            b = cache.get("baseline", {}).get(f"BT-v{v}|vs{ov}")
            if b is not None:
                ax.axhline(b, color=c_, ls="--", lw=0.9)
                ax.text(generations, b, f" BT-v{v}", fontsize=7, va="center", color=c_)
        ax.axhline(0.5, color="k", lw=0.5)
        ax.set_xlim(0, generations); ax.set_ylim(0, 1)
        ax.set_xlabel("generation (checkpoint)"); ax.set_ylabel("score (win 1 / draw 0.5)")
        ax.set_title(f"({'bc'[opponents.index(ov)]}) quick eval vs BT-v{ov}  "
                     f"(n={cache.get('n_seeds', '?')} seeds x 2 sides)", fontsize=9)
        ax.grid(alpha=0.3); ax.legend(fontsize=6, ncol=2, loc="upper left")

    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130); plt.close(fig)
    return buf.getvalue()


# ------------------------------------------------------------ HTML
def _status_rows(runs: dict, generations: int, cache: dict) -> str:
    rows = []
    for t in sorted(runs):
        r = runs[t]
        if not r["gen"]:
            rows.append(f"<tr><td>{t}</td><td colspan=6>로그 없음</td></tr>"); continue
        g, el = r["gen"][-1], r["elapsed"][-1]
        rate = el / max(g, 1)
        eta = (generations - g) * rate
        state = "완료" if (r["done"] or g >= generations) else "진행 중"
        last_ck = [c for c in cache.get("ckpt", {}).values() if c["tag"] == t]
        last_ck = max(last_ck, key=lambda c: c["gen"]) if last_ck else None
        qs = ", ".join(f"{k.replace('|vs', ' vs BT-v').replace('a', 'α=')}: {v:.2f}"
                       for k, v in (last_ck or {}).items() if k.startswith("a")) if last_ck else "–"
        rows.append(
            f"<tr><td>{t}</td><td>{state}</td><td>{g}/{generations}</td>"
            f"<td>{el/60:.1f} 분</td><td>{rate:.1f} s/gen</td>"
            f"<td>{eta/60:.0f} 분</td>"
            f"<td>{r['fit_mean'][-1]:.2f} / {r['fit_max'][-1]:.2f}</td>"
            f"<td>gen {last_ck['gen'] if last_ck else '–'}: {qs}</td></tr>")
    return "\n".join(rows)


def write_html(esdir: str, runs: dict, cache: dict, generations: int, refresh: int) -> str:
    png = render_figure(runs, cache, generations)
    with open(os.path.join(esdir, "monitor.png"), "wb") as f:
        f.write(png)
    b64 = base64.b64encode(png).decode()
    now = datetime.now().strftime("%H:%M:%S")
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="{refresh}">
<title>dfxai 학습 모니터</title>
<style>
body{{font-family:system-ui,Segoe UI,sans-serif;margin:24px;color:#222;background:#fafafa}}
h1{{font-size:18px;margin:0 0 4px}} .sub{{color:#666;font-size:12px;margin-bottom:14px}}
table{{border-collapse:collapse;font-size:13px;margin-bottom:16px;background:#fff}}
th,td{{border:1px solid #ddd;padding:6px 10px;text-align:left}} th{{background:#f0f0f0}}
img{{max-width:100%;border:1px solid #ddd;background:#fff}}
.note{{font-size:12px;color:#555;max-width:900px;line-height:1.5}}
</style></head><body>
<h1>dfxai ES 학습 모니터</h1>
<div class="sub">갱신 {now} · {refresh}초마다 자동 새로고침 · 폴더 {esdir}</div>
<table><tr><th>시드</th><th>상태</th><th>세대</th><th>경과</th><th>속도</th><th>남은 시간</th>
<th>적합도 mean / max</th><th>최근 체크포인트 빠른 평가 (점수)</th></tr>
{_status_rows(runs, generations, cache)}
</table>
<img src="data:image/png;base64,{b64}" alt="learning curves">
<p class="note">읽는 법 — (a)는 보상 셰이핑이 섞인 적합도라 방향만 보십시오. 학습이 실제로 되는지는
(b)(c)의 점수로 판단합니다: 0.5가 동률선, 점선이 BT-v1·BT-v3가 같은 상대에게 받는 점수입니다.
α=1.00은 순수 학습 정책, α=0.50은 BT-v2와 반반 혼합입니다. 빠른 평가는 20시드라 ±0.1 정도는 잡음입니다.</p>
</body></html>"""
    path = os.path.join(esdir, "monitor.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def main():
    ap = argparse.ArgumentParser(description="ES 학습 진행 뷰어")
    ap.add_argument("--esdir", default="results/es")
    ap.add_argument("--generations", type=int, default=400)
    ap.add_argument("--n-seeds", type=int, default=20, help="체크포인트 빠른 평가 시드 수")
    ap.add_argument("--watch", type=int, default=0, help="N초마다 갱신 (0이면 한 번만)")
    ap.add_argument("--no-eval", action="store_true", help="체크포인트 평가 생략(적합도만)")
    a = ap.parse_args()
    while True:
        runs = read_progress(a.esdir)
        cache = {"ckpt": {}, "baseline": {}} if a.no_eval else update_cache(a.esdir, a.n_seeds)
        p = write_html(a.esdir, runs, cache, a.generations, max(5, a.watch or 30))
        print(f"[{datetime.now():%H:%M:%S}] {p}  ({len(cache.get('ckpt', {}))} ckpt evaluated)", flush=True)
        if not a.watch:
            break
        time.sleep(a.watch)


if __name__ == "__main__":
    main()
