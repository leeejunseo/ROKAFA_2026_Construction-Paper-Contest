#!/bin/bash
# 확장 하이브리드(RES·GATE) 학습이 끝나기를 기다렸다가 평가·분석까지 자동으로 이어 간다.
#   1) 학습 체크포인트 6개(RES·GATE 시드 0·1·2 의 200세대)가 모두 있는지 확인. 없으면 안내하고 종료
#   2) 붕괴 진단 요약 (|δ|, g 표준편차) → results/main_hyb/diag.txt
#   3) 본평가: BT 3종 + 학습 정책 시드 11(순수·선형 혼합 α=0.5·Shield) + RES 6 + GATE 6
#      = 18조건 × 400교전 → results/main_hyb
#   4) 리포트·그림·자동 문서 → paper/results_hyb.md
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")/.."
LOG=results/eval_hybrid_ext.log
echo "=== 시작 $(date)" > "$LOG"
MISSING=0
for d in es_res es_gate; do for s in 0 1 2; do
  f=results/$d/ckpt_seed${s}_gen00200.npz
  if [ ! -f "$f" ]; then echo "없음: $f  (paper/다음_할_일.md 1단계로 먼저 학습)" | tee -a "$LOG"; MISSING=1; fi
done; done
[ $MISSING -eq 1 ] && exit 1
echo "=== 체크포인트 6개 확인 $(date)" >> "$LOG"
mkdir -p results/main_hyb
python - <<'EOF' > results/main_hyb/diag.txt
import json, glob, os
print("계열 | 시드 | 세대 | 진단값 평균 (첫 세대 → 25 → 100 → 200) | 표준편차(200) | 최종 적합도")
for kind, d, lab in (("RES", "results/es_res", "|δ|"), ("GATE", "results/es_gate", "g")):
    for f in sorted(glob.glob(f"{d}/history_seed*.json")):
        h = json.load(open(f, encoding="utf-8"))
        g = {r["gen"]: r for r in h}
        pick = [g[k] for k in (1, 25, 100, 200) if k in g]
        print(kind, os.path.basename(f), len(h), lab,
              " → ".join(f"{r['diag_mean']:.3f}" for r in pick),
              f"std {h[-1]['diag_std']:.3f}", f"fit {h[-1]['fit_theta']:.2f}")
EOF
cat results/main_hyb/diag.txt >> "$LOG"
RES=""; GATE=""
for s in 0 1 2; do for g in 100 200; do
  RES="$RES results/es_res/ckpt_seed${s}_gen00${g}.npz"
  GATE="$GATE results/es_gate/ckpt_seed${s}_gen00${g}.npz"
done; done
echo "=== 본평가 시작 $(date)" >> "$LOG"
python -m dfxai.experiments.run_eval \
    --ckpts results/es/ckpt_seed11_gen00300.npz --alphas 0.5 1.0 \
    --shield-ckpts results/es/ckpt_seed11_gen00300.npz \
    --residual-ckpts $RES --gate-ckpts $GATE \
    --opponents 2 3 --n-seeds 100 --workers 10 --outdir results/main_hyb >> "$LOG" 2>&1
echo "=== 리포트 시작 $(date)" >> "$LOG"
python -m dfxai.analysis.report  --outdir results/main_hyb >> "$LOG" 2>&1
python -m dfxai.analysis.figures --outdir results/main_hyb --esdir results/es_res >> "$LOG" 2>&1
python -m dfxai.analysis.paper   --outdir results/main_hyb --esdir results/es_res --out paper/results_hyb.md >> "$LOG" 2>&1
echo "=== 전부 완료 $(date)" >> "$LOG"
