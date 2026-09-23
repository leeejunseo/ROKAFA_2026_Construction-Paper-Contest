#!/bin/bash
# 잔차형(RES)·게이팅형(GATE) 혼합 학습 — 하이브리드_확장_실험설계.md 4절
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")/.."
for s in 0 1 2; do
  echo "=== RES seed $s $(date)"
  python -m dfxai.rl.train_es_hybrid --kind residual --beta 0.35 --generations 200 --pop 32 --episodes 8 \
      --sigma 0.05 --lr 0.01 --workers 10 --checkpoint-every 25 --seed $s --tag seed$s --outdir results/es_res
done
for s in 0 1 2; do
  echo "=== GATE seed $s $(date)"
  python -m dfxai.rl.train_es_hybrid --kind gating --rl-ckpt results/es/ckpt_seed11_gen00300.npz --generations 200 --pop 32 --episodes 8 \
      --sigma 0.05 --lr 0.01 --workers 10 --checkpoint-every 25 --seed $s --tag seed$s --outdir results/es_gate
done
echo "=== DONE $(date)"
