# C안 엔지니어 인수인계서 (2026-11-06 기준)

> 작성: 이준서(엔지니어, E1–E10). 이 문서 하나로 엔지니어 일을 이어받을 수 있게 썼다.
> 근거와 수치는 `paper/C안_엔지니어_진행.md`, 팀원이 자기 PC 로 학습을 돌리는 방법은 `paper/C안_분산학습_안내.md`,
> 역할·판정일·결정 사항은 `C안_실행명세서.md`(박윤준)에 있다.

## 1. 지금 상태

* **돌고 있는 작업은 없다.** 11/6 수용 검사·시범 학습·재현 시험이 모두 끝났다.
* **11/4 작업은 커밋·푸시했다(`ff4c15f`). 손형래 11/5 전달분 반영은 아직 커밋하지 않았다.** 이 PC 의 작업 폴더에만 있다(6절). 회의 확정을 전달받지 못해 커밋과 태그를 미뤘다.
* 가장 급한 일은 **회의 결과 확인과 11/8 G2 판정**이다. G1 은 ④ 행동 복제 R²(3DOF 0.877, 6DOF 0.831)만 미달이다. G2 시범은 규칙으로는 통과지만 근거가 약하다(`paper/C안_엔지니어_진행.md` 4절 끝).
* **저장소가 공개 상태다.** 푸시 전에 3절의 교범 문구 항목을 먼저 정한다.

| ID | 할 일 | 상태 | 남은 일 |
|---|---|---|---|
| E1 | 학습기 6DOF 옵션 | 끝 | 없음 |
| E2 | 이어하기(`--resume`) | 끝 | 없음 |
| E3 | 6DOF 과G 측정 | 끝(측정 정의 수정) | 없음 |
| E4 | 새 BT 통합 | BT-M 연결 끝, 손형래 11/5 전달분 반영(커밋 전) | 회의 확정 후 커밋 |
| E5 | 6DOF 시범 학습 | 새 규칙 시범 1회: 규칙으로 통과(50세대 +0.056, 구간 +0.009 ~ +0.104), 근거 약함 | 시드 하나 더 돌려 재현 확인 → 11/8 G2 |
| E6 | 분산 실행 키트 | 끝(축소 설정으로 끝까지 시험) | 태그 달기, 팀원 실측 시간으로 배정표 만들기 |
| E7 | 사전 판정 기준 | 초안, 11/5 제안을 "(제안)"으로 반영 | D1·D2·D3·D5·D7 확정, "(제안)" 삭제 → 학습 전에 커밋 |
| E8 | 평가 파이프라인 | 끝(시험 통과), 원래 원추 재평가 추가 | 11/18–20 실제 실행 |
| E9 | 보험 2 (3DOF) | 명령 준비 | G1 확정과 코드 커밋 직후 시작 |
| E10 | JSBSim 생성 시간 | 측정만 함(교전 시간의 5%), 개선 안 함 | 없음 |

## 2. 날짜별로 할 일

| 언제 | 할 일 | 누가 | 방법 |
|---|---|---|---|
| 지금 | **회의 결과 확인.** 11/5 제안(상대 교체, 기총 원추, G1 ③ 기준), G1 ④ 처리, D1–D7 이 확정됐는지 본다. 확정되지 않은 항목은 되돌린다 | 엔지니어, 팀(박윤준 진행) | 명세서 5절, 이 문서 3절 |
| 회의 확정 직후 | 코드 커밋(6절) → **E9 보험 2 시작** | 엔지니어 | 4.4절 |
| 11/6–8 | 새 규칙 시범 시드 1 (이 PC 로 약 1시간) → 판정 | 엔지니어 | 4.3절 |
| 11/8 | **G2 판정.** 시범 두 개가 모두 오르면 6DOF 학습 진행, 아니면 명세서의 실패 경로(6DOF 학습 포기 → 3DOF 보험 2 가 본 결과 + 6DOF 평가) | 팀 | 두 시범의 `pilot_eval.md`, `paper/C안_엔지니어_진행.md` 4절 끝 |
| 11/8 전 | `plan.json`·`paper/C안_사전판정기준.md` 확정 → 재현 시험 → 커밋 → 태그 `c-plan-v1` → 푸시 | 엔지니어 | 4.1절, 6절 |
| 11/8 | 팀원 전원 `selftest` → 지문이 모두 같은지 확인, 시험 시간으로 배정표를 다시 만들어 드라이브에 올림 | 전원, 엔지니어 | `paper/C안_분산학습_안내.md` |
| 11/9 (늦어도 11/10) | **G3 본 학습 시작.** 11/10 을 넘기면 10시드 × 200세대로 줄인다 | 전원 | 각자 `job queue` |
| 11/9–17 | 수시로 결과 수집·검사 | 엔지니어 | 4.5절 |
| 11/17 24:00 | **G4 학습 동결.** 그때까지 끝난 시드만 쓴다 | — | — |
| 11/18–20 | 평가(상대별·상호 대전·D\*) | 엔지니어 | 4.6절 |
| 11/21–23 | 3.1절(실험 환경)·3.3절(결합 방식과 학습)·부록 집필 | 이준서 | 명세서 6절 |

## 3. 팀이 정해야 할 것 (정해지면 엔지니어가 반영)

| 결정 | 지금 들어가 있는 값(제안) | 반영할 곳 |
|---|---|---|
| G1 ④ 기준 | —(3DOF 0.877, 6DOF 0.831 로 둘 다 0.9 미달) | `paper/C안_사전판정기준.md` |
| G1 ③ 기준 | 학습 상대끼리 0.35–0.65 (손형래 제안) | `accept_bt.py`, 사전판정기준 |
| D1 학습·보류 상대 | 학습: BT-M + BT-E. 보류: BT-R + BT-M+ (손형래 제안). 보류 상대가 BT-M 과 조종 논리가 거의 같아, BT-v1·v3 을 평가 전용으로 남기는 안을 권고 | `plan.json` 의 `train_opponents`·`heldout_opponents` |
| 6DOF 기총 원추 | 6°→12° (손형래 제안) | `plan.json` 의 `gun_cone_6dof` |
| 하드덱 침범의 해석 | 위반율 보고만 함. 시범에서 침범 교전을 패로 치면 상승이 사라졌다 | 사전판정기준 |
| 적합도 대안(K_WEZ 0.5 → 1.0) | 쓰지 않음(G2 규칙 통과). 쓰려면 새 항목이 필요하므로 태그 전에 정한다 | `dfxai/rl/shaping.py` |
| D2 교전 규칙 | 120초, 시간종료 때 체력 비교(유지) | 사전판정기준 |
| D3 주 지표 | 일반화 점수(보류 상대 + 상호 대전 평균) | 사전판정기준 |
| D4 관측 추가 | 필요 없음(BT-M 은 기존 관측 20개로 돈다) | — |
| D5 보험 2 의 롤 속도 상한 | 없음(`null`). 쓰기로 하면 180 | `plan.json` 의 `roll_rate_max_3dof` |
| D6 PC 별 24시간 가동 | — | 배정표 |
| D7 게이팅형 짝 | 같은 시드 N 의 초기화형(이미 반영) | `plan.json` 의 `gating.needs` |
| 교범 PDF 를 저장소에 둘지 | git 에서 제외 | `.gitignore` |
| 교범 문구·성능표 수치 | 저장소가 공개인데 `bt_manual.py`·대응표에 들어 있고 일부는 이미 푸시됐다. 다음 푸시 전에 저장소를 비공개로 돌릴지, 표현을 줄일지 정한다(대응표 1절 보안 확인) | 저장소 설정, 손형래 |
| 규모 | 20시드 × 300세대 | `plan.json` 의 `seeds`·`generations` |

## 4. 명령 모음

저장소 최상위 폴더의 PowerShell 에서 친다. 이 PC 는 i5-1235U 노트북이라 `--workers 11` 을 쓴다. 다른 PC 는 코어 수에 맞춰 줄인다.

### 4.1 코드를 고쳤을 때: 재현 시험

```powershell
python -m dfxai.cplan.repro_check --workers 11 --sixdof-resume
```

7개 항목이 모두 "통과"여야 한다. 결과는 `results/cplan/checks/repro_check.json` 에 남는다. 기존 3DOF 학습 4종의 기록, 본실험 교전 180개, 3DOF·6DOF 이어하기를 확인한다. 이 PC 에서 1시간 이내다.

### 4.2 기준선 BT 수용 검사 (G1)

```powershell
python -m dfxai.cplan.accept_bt --bt btm --workers 11
```

이 PC 로 약 30분 걸린다. 결과는 `results/cplan/accept_btm.md` 에 남는다. 부산물로 초기화형 학습의 출발점 `results/cplan/bc/bc_btm_6dof.npz`·`bc_btm_3dof.npz` 를 다시 만든다(행동 복제 1000회 반복, 수집 상대는 학습 상대). **다시 돌리면 출발점 파일을 덮어쓰므로 태그를 단 뒤에는 돌리지 않는다.**

손형래의 검증 스크립트(`gunrule.py`, `validate_opponents.py`)는 먼저 `$env:PYTHONIOENCODING = "utf-8"` 을 친 뒤 돌린다(cmd 에서는 `set PYTHONIOENCODING=utf-8`). `validate_opponents.py` 는 옛 원추로 고정돼 있다.

### 4.3 6DOF 시범 학습과 판정 (G2)

```powershell
python -m dfxai.rl.train_es --generations 50 --pop 32 --episodes 8 --sigma 0.05 --lr 0.01 --workers 11 --checkpoint-every 10 --seed 1 --tag seed1 --init results/cplan/bc/bc_btm_6dof.npz --bt btm --opponents btm bte --sixdof --cone 6 12 --resume --outdir results/cplan/pilots/6dof_btm_c612_seed1
python -m dfxai.cplan.pilot_eval --dir results/cplan/pilots/6dof_btm_c612_seed1 --bc results/cplan/bc/bc_btm_6dof.npz --n-seeds 50 --workers 11
```

학습은 1세대에 약 60초라 50세대에 약 50분, 판정은 약 10분이다. PC 가 꺼지면 첫 명령을 그대로 다시 치면 이어서 돈다.
판정 상대와 기총 원추는 `plan.json` 에서 읽는다. 판정 규칙은 시드 0 시범과 같다. 마지막 체크포인트(50세대)의 출발점 대비 차이의 95% 구간 하한이 0 보다 커야 한다.
`pilot_eval.md` 를 볼 때는 학습 상대·보류 상대 점수와 하드덱 열을 함께 본다. 시드 0 의 해석은 `paper/C안_엔지니어_진행.md` 4절 끝에 있다.

### 4.4 보험 2 (E9, G1 통과 직후)

```powershell
python -m dfxai.cplan.admin assign --pcs PC1:이준서:11 --env 3dof --out results/cplan/배정표_3dof.csv
python -m dfxai.cplan.job queue --csv results/cplan/배정표_3dof.csv --pc PC1 --name 이준서 --env 3dof
```

* 80작업 × 약 30분 ≈ 40시간이다. 명세서는 손형래 PC 도 보험 2 에 쓰게 되어 있다. 두 PC 로 나누려면 `--pcs PC1:이준서:11 PC2:손형래:<작업자 수>` 로 배정표를 만들고 각 PC 에서 자기 `--pc` 로 돌린다.
* 시작 전에 코드를 커밋한다. 결과 파일에 커밋 해시가 남으므로, 커밋하지 않은 코드로 돌리면 수집 검사에서 "코드 수정본으로 학습"으로 걸린다.
* D5 로 롤 속도 상한을 쓰기로 하면 먼저 `plan.json` 의 `roll_rate_max_3dof` 를 180 으로 바꾼다.

### 4.5 분산 학습 운영

```powershell
# 배정표 (팀원 selftest 의 시험 시간을 PC 마다 네 번째 칸에 넣는다)
python -m dfxai.cplan.admin assign --pcs PC1:이준서:11:<시험초> PC2:손형래:3:<시험초> PC3:박윤준:3:<시험초> PC4:이성재:3:<시험초> PC5:뚜언닷:3:<시험초> --out 배정표.csv
# 수집·검사 → results/cplan/collect_6dof.md
python -m dfxai.cplan.admin collect --from "G:\내 드라이브\C안_학습결과" --env 6dof
```

수집 검사는 300세대까지 끝났는지, 결과에 적힌 커밋이 태그와 같은지, 200세대 체크포인트가 있는지, 설정이 `plan.json` 과 같은지를 본다.
팀원이 겪을 문제와 대처는 `paper/C안_분산학습_안내.md` 6절에 있다.

### 4.6 평가 (G4 이후)

```powershell
python -m dfxai.cplan.evaluate all --env 6dof --workers 11
```

결과 문서는 `paper/results_cplan_6dof.md`, 정책별 값은 `results/cplan/eval_6dof/per_policy_b300.csv`(박윤준 Y4 분석의 입력)다.
`all` 은 원래 원추(3°→10°) 재평가까지 하므로 이 PC 로 약 24시간, 200세대 체크포인트까지 같은 방식으로 하면 약 이틀 걸린다. 11/18–20 일정이 빠듯하므로 평가도 PC 를 나눌 준비를 한다.

## 5. 파일 지도

### 5.1 코드 (C안에서 새로 만들거나 고친 것)

| 파일 | 하는 일 |
|---|---|
| `dfxai/cplan/plan.json` | 본 학습 설정 전부(기준선·상대·기총 원추·시드·세대·σ·lr). 팀원은 숫자를 치지 않고 이 파일을 쓴다 |
| `dfxai/cplan/job.py` | 팀원 실행기: `selftest`, `queue`, 결과 압축 |
| `dfxai/cplan/admin.py` | 엔지니어용: 배정표(`assign`), 수집·검사(`collect`) |
| `dfxai/cplan/accept_bt.py` | G1 수용 검사 + 행동 복제 출발점 생성 |
| `dfxai/cplan/pilot_eval.py` | G2 시범 판정 |
| `dfxai/cplan/evaluate.py` | E8 평가, 상호 대전, D\*, 원래 원추 재평가(`orig`), 결과 문서 |
| `dfxai/cplan/repro_check.py` | 재현 시험 |
| `dfxai/agents/registry.py` | 행동트리를 이름으로 부른다(`v1`·`v2`·`v3`·`btm`·`btmplus`·`bte`·`btr`·`school`) |
| `dfxai/agents/bt_manual.py` | BT-M 원본(손형래, 11/5 판은 통과 회피 수정 포함). **고치지 않는다** |
| `dfxai/agents/bt_energy.py`·`bt_radius.py`·`bt_opponents_adapter.py` | 상대 BT-E·BT-R 원본과 어댑터(손형래, 11/5). **고치지 않는다** |
| `dfxai/agents/bt_manual_adapter.py` | BT-M 을 우리 형식에 맞춘다: 안전 노드, 규칙 수, 최적화 상수(손형래 확인 범위) |
| `dfxai/agents/bt_template.py` | 새 BT 를 만들 때의 형식 예시 |
| `dfxai/agents/shield.py`·`residual.py`·`gating.py`·`bt.py` | 하이브리드가 어느 기준선 위에서든 만들어지게 고쳤다 |
| `dfxai/rl/train_es.py`·`train_es_hybrid.py`·`train_es_bt.py`·`es_runtime.py` | 6DOF 옵션, 이어하기, 실행 기록(`meta_seed<N>.json`) |
| `dfxai/rl/pretrain_bc.py` | 행동 복제(교사·상대를 이름으로 고름, 6DOF) |
| `dfxai/sixdof/env6dof.py`·`aircraft.py`·`overg_check.py` | E3 과G 정의 수정과 점검 |
| `dfxai/experiments/run_eval.py` | 평가 조건에 기준선 이름을 쓸 수 있게 |
| `dfxai/experiments/validate_btm.py`·`validate_opponents.py` | 손형래의 검증 스크립트(고치지 않음) |
| `gunrule.py` (최상위) | 손형래의 기총 원추 재현 스크립트. `python gunrule.py . 60` |
| `requirements-cplan.txt` | 학습 PC 의 라이브러리 버전 고정 |

### 5.2 결과 (`results/cplan/`)

| 위치 | 내용 |
|---|---|
| `accept_btm.md`·`.json` | BT-M 수용 검사(G1) |
| `accept_v2.md`·`.json` | BT-v2 로 미리 돌린 수용 검사(B안 대비) |
| `bc/bc_btm_6dof.npz`·`bc_btm_3dof.npz` | **초기화형 학습의 출발점**(1000회 반복, 수집 상대 BT-M·BT-E, 6DOF 는 원추 6°→12°). 배포 태그에 넣는다 |
| `bc/bc_v2_6dof.npz`·`bc_v2_3dof.npz` | BT-v2 출발점(B안 대비) |
| `bc/*_data.npz` | 행동 복제 수집 자료. 91 MB 라 git 에서 제외했다. 출발점을 다시 맞출 때 쓴다 |
| `pilots/6dof_btm_c612_seed0/` | **새 규칙 BT-M 시범(G2 자료).** `btm_same_seeds/` 는 같은 시드로 잰 BT-M 자신의 점수 |
| `pilots/6dof_btm_seed0/` | 옛 규칙 BT-M 시범(경계선). 상대·원추·출발점이 달라 G2 자료가 아니다 |
| `pilots/6dof_v2_init/`·`6dof_v2_fresh/`·`6dof_v2_ep16/` | BT-v2 대체 시범 1·2·3(모두 하락) |
| `checks/` | 재현 시험과 수용 검사의 화면 기록 |

시범 폴더마다 `train.log`(학습 화면 기록), `history_seed0.json`, 10세대마다의 체크포인트, `meta_seed0.json`, 판정 결과 `pilot_eval.md` 가 있다.
이어하기용 `state_seed0.*` 는 git 에서 제외했다.

그 밖의 C안 결과는 `results/btm_g1/`(손형래의 G1 검증, `BT-M_recheck/` 는 torch 재실행, `BT-M_v2/` 는 11/5 판), `results/opponents/`(손형래의 상대 검증), `results/sixdof/overg_check.csv`(E3)다.

앞으로 생길 것은 다음 위치에 생긴다.

* `results/es6_<종류>/`·`results/es3_<종류>/`: 학습 결과. git 에 올리지 않고 드라이브로만 모은다.
* `results/cplan/collect_6dof.md`: 수집 검사 결과.
* `results/cplan/eval_6dof/`, `paper/results_cplan_6dof.md`: 평가 결과.

### 5.3 문서

| 파일 | 내용 |
|---|---|
| `C안_실행명세서.md` (최상위) | 박윤준 작성. 역할, 판정일, 결정 사항. 박윤준이 고쳐 다시 커밋하므로 위치를 옮기지 않았다 |
| `paper/C안_인수인계서.md` | 이 문서 |
| `paper/C안_엔지니어_진행.md` | E1–E10 의 근거와 수치 전부 |
| `paper/C안_분산학습_안내.md` | 팀원용 실행 방법 |
| `paper/C안_사전판정기준.md` | E7 초안 |
| `docs/README_H1_BT-M.md`·`docs/README_상대BT_전달.md`·`docs/BT-M_교범대응표.md` | 손형래의 전달 메모(11/4, 11/5)와 교범 대응표 |
| `docs/Basic-Employment-Manual-F-16C-RoKAF_2005.pdf` | 교범 원문. 보안성 검토 전까지 git 에서 제외했다 |

그 밖의 `paper/*.md` 와 `results/` 의 나머지 폴더는 10월 최종본(보험 1)의 자료다. 논문 근거라서 손대지 않았다.

## 6. 커밋 상태와 주의

**이미 커밋·푸시한 것:** 11/4 C안 작업과 그 전 작업 전부(`ff4c15f`, `origin/main`).

**아직 커밋하지 않은 것** (`git status` 로 본다)

* 손형래 11/5 전달분 반영 전부: 5절의 코드, 새 출발점 `bc_btm_*.npz`, 수용 검사·시범·재현 시험 결과, 문서, `gunrule.py`.
* 회의에서 바뀐 항목이 있으면 커밋 전에 그 항목을 되돌리고 4.1·4.2절을 다시 돌린다.

**git 에 올리지 않는 것** (`.gitignore` 에 있다)

* `경진대회 안내문/`, 교범 PDF.
* `results/es6_*`·`results/es3_*`·`upload/`: 학습 결과는 드라이브로만 모은다.
* `state_*`(이어하기 파일), `results/cplan/bc/*_data.npz`.

**커밋 순서**

1. 재현 시험(4.1)을 돌려 모두 통과하는지 확인한다(11/6 지금 코드로 통과).
2. 코드·문서·출발점을 커밋하고 푸시한다. 이렇게 해 두면 다른 PC 에서도 이어받을 수 있다. **저장소가 공개이므로 푸시 전에 3절의 교범 문구 항목을 먼저 정한다.**
3. `plan.json` 과 사전 판정 기준이 확정되면 `plan.json` 의 `"tag"` 에 태그 이름을 적고 커밋한다. 그 커밋에 태그를 달고 `git push origin main --tags` 로 올린다.
4. **태그를 단 뒤에는 코드를 고치지 않는다.** 버그로 고쳐야 하면 새 태그를 단다. 그 전에 돌린 작업을 다시 할지는 팀이 정한다.

## 7. 알아 둘 것

* **6DOF 는 느리다.** 이 PC 에서 1세대 약 60초, 본 학습 작업 하나(300세대) 약 5시간이다. 명세서 가정의 약 2배다. 팀원 PC 의 실측값(`selftest`)으로 배정표를 다시 만든다.
* **팀원 학습은 반드시 `job.py` 로 돌린다.** 실행기가 결정성 설정(BLAS 커널·SIMD 경로 고정)과 커밋 해시 기록을 맡는다. 학습기를 직접 부르면 PC 마다 결과가 다를 수 있다.
* **행동 복제는 1000회 반복이 기준이다.** 11/6 판 출발점만 새 규칙에 맞는다. 옛 규칙 시범(`6dof_btm_seed0`)은 옛 판에서 출발했다.
* **행동 복제 R² 는 수집 상대에 따라 달라진다.** 손형래의 0.909 는 BT-v1·2·3 상대 값이라 BT-M·BT-E 상대 값(3DOF 0.877, 6DOF 0.831)과 비교할 수 없다.
* **BT-v2 대체 시범으로 BT-M 결과를 예측하면 안 된다.** BT-v2 복제본은 6DOF 에서 과G 를 많이 내서 학습이 "덜 당기기"로 무너졌다. BT-M 은 6DOF 과G 가 0 이다.
* **BT-M 시범의 약점.** 원추를 넓혀 접촉은 늘었지만 교전의 93% 가 여전히 시간종료이고 격추승은 1% 미만이다. 학습할수록 하드덱 침범도 늘어 50세대에 교전의 18%, 30·40세대에 44–47% 다. 침범 교전을 패로 치면 상승이 사라진다. 감독형은 이를 막지만 초기화형·혼합형에는 남는다. 결과를 해석할 때 함께 본다.
* **출발점보다 올랐다고 BT-M 을 넘은 것은 아니다.** 복제본은 BT-M 보다 0.094 낮게 출발한다. 시범의 50세대도 BT-M 보다 0.037 낮다.
* **3DOF 와 6DOF 는 기총 원추가 다르다.** 두 환경의 비교(RQ3)에는 비행역학과 원추 차이가 함께 들어 있다.
* **감독형의 G 상한**은 기준선에서 읽는다. BT-M 기반은 7.7 G, 기존 BT-v3 기반 감독형은 8 G 다.
* **손형래의 파일(`bt_manual.py`, `bt_energy.py`, `bt_radius.py`, `validate_btm.py`, `validate_opponents.py`)은 고치지 않는다.** 형식을 맞출 일은 어댑터에서 한다.
* **E3 이후 6DOF 과G 값이 달라졌다.** 10/1 교차검증의 과G(참고값)는 다시 돌리면 바뀐다. 판정 C4 는 하드덱만 썼으므로 판정은 그대로다.
* 학습을 오래 돌릴 때는 PC 전원을 연결하고 절전 모드를 끈다. 꺼져도 같은 명령으로 이어진다.

## 8. 함께 일하는 사람

| 사람 | 맡은 것 | 엔지니어와 주고받을 것 |
|---|---|---|
| 박윤준 | 총괄, 판정 회의, 지표(Y3), 분석(Y4) | G1–G4 판정, D1–D7, 평가 결과 CSV |
| 손형래 | BT-M·상대 BT(H1–H6), 보험 2 용 PC | BT-E 강도(시범 결과 전달), 교범 문구, 진행 문서 3.4절 |
| 이성재, 뚜언닷 | 문헌, 영문, 연산 운영 | PC 제공, `selftest` 결과 |
