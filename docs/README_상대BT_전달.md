# 교범 기반 학습·평가 상대 전달 메모 — 손형래 → 이준서 (2026-11-05)

## 1. 바뀌는 상대 구성 (D1 제안)

| 역할 | 이전 제안 | 새 제안 |
|---|---|---|
| 학습 상대 | BT-M, BT-v2 | **BT-M(선회율 싸움형), BT-E(에너지·수직형)** |
| 평가 전용(보류) 상대 | BT-v1, BT-v3, 교내대회 BT | **BT-R(반경 싸움형), BT-M+(교범 상세판)** |

BT-v1·v2·v3 은 쓰지 않는다. 네 상대 모두 같은 교범(4.3~4.6절)의 서로 다른 게임플랜이고, 규칙마다 절 번호가 `docs/BT-M_교범대응표.md` 6절에 있다.
6자유도 본 학습과 3자유도 보험 2 모두 같은 구성으로 한다.

## 2. 파일

| 파일 | 내용 | 비고 |
|---|---|---|
| `dfxai/agents/bt_energy.py` | BT-E (25노드) | 새 파일 |
| `dfxai/agents/bt_radius.py` | BT-R (15노드) | 새 파일 |
| `dfxai/agents/bt_opponents_adapter.py` | registry 형식(`BTEPolicy`, `BTRPolicy`, `rule_count`) | 새 파일. 상대 전용이라 theta 는 받지 않는다 |
| `dfxai/agents/registry.py` | `bte`, `btr` 등록 (`registry_bte_btr.diff`) | **네 파일이라 diff 로 보냄.** 확인하고 적용해 줘 |
| `dfxai/agents/bt_manual.py` | 통과 회피 수정(아래 3절) | 내 파일 |
| `dfxai/experiments/validate_opponents.py` | 4종 상호 검증(6쌍 × 400교전 × 3·6DOF) | 새 파일 |
| `results/opponents/` | 상호 검증 결과 | |
| `results/btm_g1/BT-M_v2/` | 수정한 BT-M 의 G1 재검증 | |

## 3. BT-M 도 조금 바뀌었다 (중요)

6DOF 상호 검증에서 정면 통과 때 공중충돌이 기준을 넘었다(BT-R 대 BT-M+ 13.5%, BT-M 대 BT-M+ 10.0%).
원인은 통과 회피(`lateral_offset`)가 배면에 가까운 자세에서 바로 6 G 로 당겨, 기수가 상대 고도 쪽으로 떨어진 것이다.
교범 4.4.5.2 의 "unloaded roll to set Lv" 대로, 실제 뱅크가 회피 뱅크에서 30° 안에 들어오기 전에는 1 G 로 굴리기만 하도록 고쳤다
(`BTMParams.pass_roll_tol`). 충돌은 7.5% 이하로 내려갔다. 400교전 재검증(`results/btm_g1/BT-M_v2/`)에서도 BT-v2 상대 3DOF 0.478, 6DOF 0.463, 행동 복제 R² 0.909(sklearn)로 G1 을 모두 충족했다(수정 전 0.472, 0.465).

**그래서 `accept_bt --bt btm` 을 다시 돌려 행동 복제 출발점(`bc_btm_*.npz`)을 새로 만들어야 한다.** 기총 원추 변경(6DOF 6°→12°)도 같이 들어가니 한 번에 하면 된다.

## 4. 부탁할 것

1. `registry_bte_btr.diff` 적용 (`bte`, `btr`)
2. `plan.json` 의 학습 상대 `["btm", "bte"]`, 보류 상대 `["btr", "btmplus"]` (6DOF·3DOF 둘 다)
3. `accept_bt.py` 의 G1 기준 "BT-v2 상대 ≥ 0.45" → "학습 상대끼리 점수 0.35~0.65" (BT-M 대 BT-E 결과는 `results/opponents/report.md`)
4. 6DOF 기총 원추 6°→12° (사거리 1,200 m 그대로)
5. `accept_bt --bt btm` 재실행 → 새 출발점
6. 새 상대 + 새 원추로 BT-M 6DOF 시범 1회 → G2 자료
7. `evaluate.py` 평가 상대 변경, 보조 지표(HP 차이, WEZ 체류 시간) 추가, 원래 원추(3°→10°)로도 한 번 더 평가
8. 사전판정기준 문서 반영

검증 다시 돌리기:

```powershell
set PYTHONIOENCODING=utf-8
python -m dfxai.experiments.validate_opponents --workers 11
python -m dfxai.experiments.validate_btm --variant core --workers 11 --outdir results/btm_g1/BT-M_v2
```
