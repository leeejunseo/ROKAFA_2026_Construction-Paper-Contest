#!/bin/bash
# 최종 파이프라인: 시드 8~19 학습과 8시드 본실험 파이프라인이 끝난 뒤 실행된다.
#   1) 상수 최적화 BT(BTO) 3시드 학습      2) 밑바닥 PPO 2시드 학습
#   3) 20시드 본실험 + BTO + 방패 혼합 + PPO 평가 → results/main (8시드는 main_8seeds 로 보관)
#   4) 보고서·그림·results_auto.md          5) 25세대 간격 예산 평가 (results/dense)
#   6) 규칙 민감도 (results/sens_t240, sens_draw)  7) 정책 풀 (results/pool)
#   8) 공통 상태 D*, 강건성 문서 (paper/results_ext.md)  9) 학습 뷰어
cd "/c/Users/User/Desktop/공군사관학교/생도대/대회/공사생도논문경진대회" || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
LOG=results/final_pipeline.log
step() { echo "[$(date '+%H:%M:%S')] $*" >> "$LOG"; }

step "대기: 시드 8~19 학습"
until [ -f results/es/SEEDS_8_19_DONE ]; do sleep 120; done
step "대기: 8시드 파이프라인(results_auto.md 갱신)"
until [ paper/results_auto.md -nt results/main/manifest.json ] && [ -f results/main/manifest.json ]; do sleep 120; done
sleep 600
step "시작"

# 1) BTO
for s in 0 1 2; do
  step "BTO seed $s"
  python -m dfxai.rl.train_es_bt --generations 300 --pop 32 --episodes 8 --workers 10 \
      --seed $s --tag seed$s --checkpoint-every 50 --outdir results/es_bto > results/es_bto_seed$s.log 2>&1
done

# 2) PPO from scratch
mkdir -p results/ppo
for s in 0 1; do
  step "PPO seed $s"
  python -m dfxai.rl.train_ppo --timesteps 1500000 --n-envs 10 --seed $s \
      --alpha-lo 1.0 --alpha-hi 1.0 --no-curriculum --save-every 500000 \
      --outdir results/ppo > results/ppo/stdout_seed$s.log 2>&1
done

# 3) 20시드 본실험
step "본실험 20시드"
CK=""
for g in 100 200 300; do for s in $(seq 0 19); do CK="$CK results/es/ckpt_seed${s}_gen00${g}.npz"; done; done
SHD=""; for s in $(seq 0 19); do SHD="$SHD results/es/ckpt_seed${s}_gen00300.npz"; done
BTO=""; for s in 0 1 2; do for g in 100 200 300; do BTO="$BTO results/es_bto/ckpt_seed${s}_gen00${g}.npz"; done; done
PPO=$(ls results/ppo/ckpt_seed*_step*.npz 2>/dev/null | tr '\n' ' ')
python -m dfxai.experiments.run_eval --ckpts results/es/bc_init.npz $CK \
    --alphas 0.25 0.5 0.75 1.0 --opponents 2 3 --n-seeds 100 --workers 10 \
    --bto-ckpts $BTO --shield-ckpts $SHD --ppo-ckpts $PPO \
    --outdir results/main20 > results/main20_eval.log 2>&1
if [ -f results/main20/manifest.json ]; then
  [ -d results/main_8seeds ] && rm -rf results/main_8seeds
  mv results/main results/main_8seeds && mv results/main20 results/main
  step "main ← 20시드 (8시드는 main_8seeds)"
else
  step "오류: 20시드 평가 실패 (results/main20_eval.log)"; exit 1
fi

# 4) 보고서·그림·문서
step "보고서"
python -m dfxai.analysis.report --outdir results/main >> "$LOG" 2>&1
python -m dfxai.analysis.report --outdir results/main --lang ko --figs-only >> "$LOG" 2>&1
python -m dfxai.analysis.figures --outdir results/main --esdir results/es --traj-seed 10000 --lang en >> "$LOG" 2>&1
python -m dfxai.analysis.figures --outdir results/main --esdir results/es --traj-seed 10000 --lang ko >> "$LOG" 2>&1
python -m dfxai.analysis.paper --outdir results/main --esdir results/es --out paper/results_auto.md >> "$LOG" 2>&1
python -m dfxai.viz.replay --from-run results/main --red bt:3 --outdir results/replay >> "$LOG" 2>&1

# 5) 25세대 간격 예산 (시드 8~19, 순수 학습만)
step "세밀 예산 평가"
DK=""; for s in $(seq 8 19); do for g in 025 050 075 100 125 150 175 200 225 250 275 300; do DK="$DK results/es/ckpt_seed${s}_gen00${g}.npz"; done; done
python -m dfxai.experiments.run_eval --ckpts $DK --alphas 1.0 --opponents 2 3 --n-seeds 50 --workers 10 \
    --no-bt --outdir results/dense > results/dense_eval.log 2>&1
python -m dfxai.analysis.report --outdir results/dense >> "$LOG" 2>&1
python -m dfxai.xai.common_state --outdir results/dense --workers 10 >> "$LOG" 2>&1

# 6) 민감도
step "민감도"
SK=""; for s in $(seq 0 19); do SK="$SK results/es/ckpt_seed${s}_gen00300.npz"; done
SB=""; for s in 0 1 2; do SB="$SB results/es_bto/ckpt_seed${s}_gen00300.npz"; done
python -m dfxai.experiments.run_eval --ckpts $SK --alphas 1.0 --bto-ckpts $SB --opponents 2 3 --n-seeds 100 --workers 10 \
    --episode-time 240 --outdir results/sens_t240 > results/sens_t240_eval.log 2>&1
python -m dfxai.analysis.report --outdir results/sens_t240 >> "$LOG" 2>&1
python -m dfxai.experiments.run_eval --ckpts $SK --alphas 1.0 --bto-ckpts $SB --opponents 2 3 --n-seeds 100 --workers 10 \
    --timeout-rule draw --outdir results/sens_draw > results/sens_draw_eval.log 2>&1
python -m dfxai.analysis.report --outdir results/sens_draw >> "$LOG" 2>&1

# 7) 정책 풀
step "정책 풀"
python -m dfxai.experiments.round_robin --rl-ckpts $SK --bto-ckpts $SB --n-seeds 30 --workers 10 --outdir results/pool >> "$LOG" 2>&1

# 8) 공통 상태 D* + 강건성 문서
step "공통 상태 D*"
python -m dfxai.xai.common_state --outdir results/main --workers 10 >> "$LOG" 2>&1
step "강건성 문서"
python -m dfxai.analysis.robustness --outdir results/main --out paper/results_ext.md \
    --dense-dir results/dense --pool-dir results/pool --sens-dirs results/sens_t240 results/sens_draw >> "$LOG" 2>&1

# 9) 학습 뷰어
python -m dfxai.rl.monitor --esdir results/es --generations 300 --n-seeds 20 >> "$LOG" 2>&1
step "완료"
echo FINAL_DONE > results/FINAL_DONE
