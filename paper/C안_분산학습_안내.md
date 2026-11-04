# C안 분산 학습 안내 (팀원용)

> 대상: 자기 PC로 학습을 돌리는 조원 전원. 작성 2026-11-04(이준서). 명세: `C안_실행명세서.md` 2.1절.
> 배포 태그 이름(예: `c-plan-v1`)과 공유 드라이브 폴더는 이준서가 단톡방에 따로 알린다.

## 0. 한눈에

1. **한 번만 준비한다**(11/8–9): 코드 받기 → 태그로 고정 → 파이썬 환경 만들기.
2. **결정성 확인**: 시험 명령 하나를 돌려 나온 "지문"(영문·숫자 12자리)을 배정표에 적는다. 전원 같아야 한다.
3. **학습 시작**: 명령 하나를 치고 PC를 켜 둔다. 내 PC 몫의 작업이 시드 순서대로 전부 돈다.
4. **올리기**: 작업 하나가 끝날 때마다 `upload\` 폴더에 zip이 생긴다. 공유 드라이브 `C안_학습결과/`에 올리고 배정표의 상태를 "완료"로 바꾼다.

PC가 꺼지거나 재부팅되면 **3번 명령을 그대로 다시 치면 된다.** 끝난 작업은 건너뛰고, 하다 만 작업은 마지막 세대부터 이어서 돈다.

## 1. 준비 (한 번만)

PowerShell에서 한 줄씩 실행한다. Python은 python.org의 **3.13**을 설치한다(설치 화면에서 "Add python.exe to PATH" 체크).

```powershell
cd $HOME\Desktop
git clone https://github.com/leeejunseo/ROKAFA_2026_Construction-Paper-Contest.git cplan
cd cplan
git fetch --tags
git checkout c-plan-v1            # 이준서가 알려 준 태그 이름
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1      # 실행 정책 오류가 나면: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements-cplan.txt
```

이미 저장소가 있으면 `git clone` 대신 그 폴더에서 `git fetch --tags` → `git checkout c-plan-v1` 만 하면 된다.
새 PowerShell 창을 열 때마다 `cd $HOME\Desktop\cplan` → `.\.venv\Scripts\Activate.ps1` 을 먼저 한다.

**절전 끄기** (전원 연결 상태에서 잠들지 않게):

```powershell
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
```

Windows 업데이트는 설정 → Windows 업데이트 → "1주 일시 중지"를 눌러 둔다(자동 재시작 방지). 재시작돼도 이어서 돌지만 그동안 시간이 빈다.

## 2. 결정성 확인 (약 5분)

```powershell
python -m dfxai.cplan.job selftest --env 6dof
```

마지막 줄의 `지문: xxxxxxxxxxxx` 를 배정표의 `fingerprint` 칸에 적는다. **전원 같아야 한다.** 다르면 학습을 시작하지 말고
이준서에게 화면을 보낸다(파이썬·라이브러리 버전 차이가 대부분이다).

## 3. 학습 시작

배정표(`배정표.csv`)를 공유 드라이브에서 받아 저장소 폴더에 둔다. 내 PC 아이디(예: `PC3`)는 배정표의 `pc` 열에 있다.

```powershell
python -m dfxai.cplan.job queue --csv 배정표.csv --pc PC3 --name 손형래 --upload-dir "G:\내 드라이브\C안_학습결과"
```

* `--upload-dir` 는 구글 드라이브 데스크톱 앱이 있으면 그 폴더를 주고(완료 zip이 자동으로 복사됨), 없으면 빼고 손으로 올린다.
* 기본으로 코어 하나를 남기고 전부 쓴다(`--workers 코어 수 − 1`). PC를 같이 써야 하면 `--workers 2` 처럼 줄인다(느려진다).
* 화면에 `[gen   15] fit mean=… theta=…` 가 주기적으로 찍히면 정상이다. 6DOF 작업 하나는 4코어 PC에서 7–9시간이다.
* 진행 상황 보기: `python -m dfxai.cplan.job status`

게이팅형은 같은 시드의 초기화형 결과가 있어야 돈다. 배정표는 둘을 같은 PC에 넣으므로 보통은 자동으로 이어진다.
`[대기] 게이팅형 시드 N …` 이 나오면 그 초기화형 zip을 드라이브에서 받아 `python -m dfxai.cplan.job unpack <zip 파일>` 을 하고 다시 queue 를 친다.

## 4. 올리기

작업 하나가 끝나면 `upload\<이름>_<종류>_seed<N>.zip` 이 생긴다(예: `손형래_init_seed3.zip`). 드라이브 `C안_학습결과/` 에 올리고
배정표의 `status` 를 "완료"로 바꾼다. zip 안에는 체크포인트·학습 기록·실행 기록(커밋 해시 포함)·로그가 들어 있다.

## 5. 하지 말 것

* **학습 기간(11/9–11/17)에는 `git pull` 하지 않는다.** 코드가 바뀌면 PC마다 다른 코드로 학습한 결과가 섞여 비교할 수 없다.
  버그로 코드를 고쳐야 하면 이준서가 새 태그를 알린다.
* `dfxai\` 안의 파일을 고치지 않는다. 고친 코드로 돈 결과는 수집 때 걸러진다.
* 한 PC에서 queue 를 두 개 동시에 띄우지 않는다.
* `results\` 의 학습 결과를 git에 커밋하지 않는다(드라이브로만 모은다).

## 6. 문제가 생기면

| 화면 | 할 일 |
|---|---|
| `[중단] … 같은 명령을 다시 치면 이어서 돕니다` | 같은 queue 명령을 다시 친다 |
| `… 서로 다른 세대의 것입니다` | 저장 도중 꺼진 드문 경우. 해당 `results\es6_<종류>\state_seed<N>.*` 두 파일을 지우고 다시 치면 처음부터 돈다. 이준서에게 알린다 |
| `[경고] 지금 코드가 태그 … 가 아닙니다` | `git checkout <태그>` 후 다시 |
| 그 밖의 오류 | `results\es6_<종류>\log_seed<N>.txt` 를 이준서에게 보낸다 |

---

## 엔지니어용 (이준서)

```powershell
# 0) 배포 전 점검 (기존 결과 재현·이어하기)
python -m dfxai.cplan.repro_check --workers 11 --sixdof-resume
# 1) 기준선 BT 수용 검사 (G1). 부산물로 행동 복제 초기값 results/cplan/bc/bc_<기준선>_6dof.npz·_3dof.npz 생성
python -m dfxai.cplan.accept_bt --bt btm --workers 11
# 2) plan.json 확정(baseline, train_opponents, tag, roll_rate_max_3dof) → 사전 판정 기준 확정 → 커밋 → 태그
git add dfxai/cplan/plan.json results/cplan/bc/bc_btm_6dof.npz results/cplan/bc/bc_btm_3dof.npz paper/C안_사전판정기준.md
git commit -m "C안 배포: 기준선 BT-M"; git tag c-plan-v1; git push origin main --tags
# 3) 배정표
python -m dfxai.cplan.admin assign --pcs PC1:이준서:11 PC2:손형래:3 PC3:박윤준:3 PC4:이성재:3 PC5:뚜언닷:3 --out 배정표.csv
# 4) 수집·검사 (수시로) → results/cplan/collect_6dof.md
python -m dfxai.cplan.admin collect --from "G:\내 드라이브\C안_학습결과" --env 6dof
# 5) 평가 (G4 이후) → paper/results_cplan_6dof.md
python -m dfxai.cplan.evaluate all --env 6dof --workers 11
```

태그는 plan.json 의 `"tag"` 에도 같은 이름을 적은 뒤 커밋하고 단다(순서: plan.json 수정 → 커밋 → 그 커밋에 태그).
