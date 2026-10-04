#!/bin/bash
# R9 비행역학 민감도: R7 과 같은 26개 정책(BT 3종, 300세대 순수 학습 20개, BTO 3개),
# 같은 초기조건(시드 10000-10099) x 상대 2종 x 진영 2 = 조건당 400교전.
# 한 번에 하나의 비행역학 파라미터만 바꾼다. 기준은 results/main (같은 조건을 정확히 재현함).
cd "$(dirname "$0")/.." || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
W=${WORKERS:-4}
SK=""; for s in $(seq 0 19); do SK="$SK results/es/ckpt_seed${s}_gen00300.npz"; done
SB=""; for s in 0 1 2; do SB="$SB results/es_bto/ckpt_seed${s}_gen00300.npz"; done
run() {  # $1 = 출력 이름, 나머지 = run_eval 옵션
  name=$1; shift
  echo "[$(date '+%H:%M:%S')] $name 시작 ($*)"
  python3 -m dfxai.experiments.run_eval --ckpts $SK --alphas 1.0 --bto-ckpts $SB --opponents 2 3 \
      --n-seeds 100 --workers $W "$@" --outdir results/r9_$name > results/r9_$name.log 2>&1
  python3 -m dfxai.analysis.report --outdir results/r9_$name >> results/r9_$name.log 2>&1
  echo "[$(date '+%H:%M:%S')] $name 완료"
}
run cl080   --cl-max-scale 0.8
run cl120   --cl-max-scale 1.2
run thr080  --thrust-scale 0.8
run thr120  --thrust-scale 1.2
run lag080  --lag-scale 0.8
run lag120  --lag-scale 1.2
run roll180 --roll-rate-max 180
run roll090 --roll-rate-max 90
echo "R9 전체 완료"
