# H1 새 기준선 BT(BT-M) 전달 메모 — 손형래 → 이준서

## 들어 있는 것

| 파일 | 내용 |
|---|---|
| `dfxai/agents/bt_manual.py` | `ManualBTPolicy`(BT-M, 22노드), `ManualBTPolicy.full()`(BT-M+, 32노드), 상수 `BTMParams`, 안전 하위트리 `safety_act()` |
| `dfxai/experiments/validate_btm.py` | G1 검증(3·6자유도 각 400교전 + 행동복제 R² + D\*) |
| `docs/BT-M_교범대응표.md` | 노드별 교범 절·값 대응, 옮기지 않은 것과 이유, 설계 변경 기록 |
| `results/btm_g1/BT-M/`, `results/btm_g1/BT-M+/` | G1 보고서(`G1_report.md`)와 원자료(`summary.json`) |

기존 파일은 하나도 고치지 않았다. 새 파일만 추가하면 된다(`dfxai/agents/bt.py`의 Node 클래스를 가져다 쓴다).

## 실행

```bash
python -m dfxai.experiments.validate_btm --variant core --workers 4   # BT-M, 약 3분(6자유도 포함) + 복제 2~4분
python -m dfxai.experiments.validate_btm --variant full --workers 4   # BT-M+
```

결과는 결정적이다. 같은 시드로 두 번 돌렸을 때 점수가 소수점 넷째 자리까지 같았다.

## 하이브리드에 연결할 곳 (제안)

1. **행동복제 교사.** `dfxai/rl/pretrain_bc.py`의 `collect()`에서 `BTPolicy(version=teacher)` 자리에 `ManualBTPolicy()`를 넣는다.
2. **잔차형·게이팅형의 규칙 쪽.** `residual.py`·`gating.py`에서 `BTPolicy(version=2)`를 `ManualBTPolicy()`로 바꾼다. 두 정책은 `act(obs) -> [-1,1]^3` 인터페이스가 같다.
3. **감독형(Shield).** `a, node = ManualBTPolicy().safety_act(obs)`로 쓴다. `a`가 None이 아니면 규칙 지령을 쓰고, None이면 학습 정책 지령에 G 상한(7.7 G)을 건다. 안전 노드는 `SAFETY_NODES = ("lateral_offset", "recover_lift_vector_up")`.
4. **평가 조건.** `run_eval.build_condition`에 `kind == "btm"`을 추가해 `ManualBTPolicy()`를 돌려준다. 상대 목록에 BT-M을 넣을지는 팀이 정한다(학습 상대로 쓰면 보류 상대가 바뀐다).
5. **환경이 바뀌면** 고칠 곳은 `BTMParams` 하나다. 하드덱·과G·기총 사거리는 `EngagementConfig`에서 자동으로 읽는다.

## 확인 부탁

- 6자유도 과G 위반 약 2.5%: G 상한을 6.5 G로 낮춰도 0.6% 남는다. 추종 제어기의 하중 오버슛이나 Nz 측정(기존 E3 문제)인지 봐 주세요.
- 행동복제 R²는 torch가 없어서 sklearn MLP(20-32-32-3, tanh)로 잰 값이다(0.919). `pretrain_bc.fit`으로 다시 재 주세요.
- 교범 인용 가능 여부는 지도교수님 확인 대기 중이다. 안 되면 대응표의 출처 열만 Shaw(1985)로 바꾼다.
