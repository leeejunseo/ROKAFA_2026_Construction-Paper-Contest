# BT-M(교범 기반 행동트리) G1 검증 결과 (자동 생성)

노드 22개(안전 하위트리 7개), 자율 G 제한 7.7 G. 평가: 시드 10000–10099 × 진영 교대 × 상대 BT-v2·BT-v3 = 환경당 400교전.

## 1. G1 판정

| 기준 | 값 | 판정 |
|---|---|---|
| E1 3자유도 400교전 오류 없음 | 400/400 | 충족 |
| E2 3자유도 공중충돌 ≤ 10% | 0.015 | 충족 |
| E3 3자유도 지면충돌 ≤ 5% | 0.000 | 충족 |
| E4 3자유도 BT-v2 상대 점수 ≥ 0.45 | 0.477 [0.430, 0.525] | 충족 |
| E1 6자유도 400교전 오류 없음 | 400/400 | 충족 |
| E2 6자유도 공중충돌 ≤ 10% | 0.040 | 충족 |
| E3 6자유도 지면충돌 ≤ 5% | 0.000 | 충족 |
| E4 6자유도 BT-v2 상대 점수 ≥ 0.45 | 0.463 [0.427, 0.497] | 충족 |
| E5 행동복제 R² ≥ 0.9 | 0.909 (채널별 0.967, 0.941, 0.705) | 충족 |

## 2. 환경별 요약

| 지표 | 3자유도 | 6자유도 |
|---|---|---|
| 점수 vs BT-v2 | 0.4775 | 0.4625 |
| 점수 vs BT-v3 | 0.6775 | 0.5425 |
| 공중충돌률 | 0.0150 | 0.0400 |
| 자기 추락률 | 0.0000 | 0.0000 |
| 하드덱 위반율(스텝) | 0.0000 | 0.0000 |
| 과G 위반율(스텝) | 0.0000 | 0.0000 |
| 최소이격 위반율(스텝) | 0.0064 | 0.0025 |
| 6자유도 트림 성공률 | 1.0000 | 1.0000 |
| 실행 시간 [s] | 9.0265 | 168.9506 |

## 3. 결과 유형

- 3자유도 vs BT-v2: {'timeout:무': 96, 'timeout:승': 44, 'timeout:패': 29, 'gun_kill:패': 27, 'collision:무': 1, 'gun_kill:승': 3}
- 3자유도 vs BT-v3: {'collision:무': 5, 'timeout:무': 40, 'timeout:승': 101, 'gun_kill:승': 12, 'timeout:패': 14, 'gun_kill:패': 28}
- 6자유도 vs BT-v2: {'timeout:패': 34, 'timeout:무': 140, 'collision:무': 7, 'timeout:승': 19}
- 6자유도 vs BT-v3: {'timeout:패': 30, 'timeout:무': 114, 'collision:무': 9, 'timeout:승': 46, 'gun_kill:승': 1}

## 4. 설명가능성(참고)

| 환경 | D*(0.95) | D*(0.90/0.95/0.99) | F(4) | 샘플 |
|---|---|---|---|---|
| 3자유도 | 17 | {'0.9': 5.0, '0.95': 17.0, '0.99': 17.0} | 0.884 | 31528 |
| 6자유도 | 10 | {'0.9': 4.0, '0.95': 10.0, '0.99': 17.0} | 0.911 | 34990 |

행동복제 학습기: sklearn MLPRegressor(tanh 32-32, 602회 반복) — torch 없음, 샘플 161,154개.

## 5. 노드 발동 빈도

| 노드 | 3자유도 | 6자유도 |
|---|---|---|
| rate_turn_lead_or_two_circle | 0.736 | 0.746 |
| high_g_pull | 0.125 | 0.029 |
| lead_pursuit_gun | 0.042 | 0.005 |
| break_turn_into_threat | 0.040 | 0.025 |
| turn_circle_entry | 0.030 | 0.157 |
| lateral_offset | 0.014 | 0.006 |
| recover_lift_vector_up | 0.014 | 0.032 |

## 6. 상수

```json
{
 "g_margin": 0.3,
 "bubble_r": 152.4,
 "cpa_time": 2.5,
 "pass_aa": 90.0,
 "pass_ata": 90.0,
 "pass_bank": 75.0,
 "pass_g": 6.0,
 "pass_roll_tol": 0.524,
 "floor_lookahead": 5.0,
 "floor_margin": 300.0,
 "floor_zone": 1219.2,
 "v_rate_hi": 226.355,
 "thr_fast": 0.3,
 "speed_ref_alt": 4000.0,
 "def_threat_ata": 45.0,
 "def_my_ata": 90.0,
 "def_range": 2743.2,
 "off_aa": 90.0,
 "off_ata": 90.0,
 "tc_range": 1828.8,
 "entry_g": 4.0,
 "gun_range": 1371.6,
 "gun_ata": 40.0,
 "lead_gain": 1.3,
 "use_jink": false,
 "jink_ata": 15.0,
 "jink_range_scale": 1.1,
 "jink_alt": 609.6,
 "jink_roll": 90.0,
 "jink_relax_bank": 135.0,
 "jink_relax_g": 2.0,
 "use_range_control": false,
 "range_floor": 1219.2,
 "relax_g": 4.0,
 "use_overshoot_lag": false,
 "overshoot_range": 304.8,
 "lag_gain": 0.3,
 "use_closure_idle": false,
 "idle_closure": 30.0,
 "idle_range": 914.4,
 "use_lv_bias": false,
 "tc_lv_bias": 3.0,
 "rate_lv_bias": -3.0,
 "entry_climb_bias": 3.0,
 "use_entry_speed": false,
 "entry_v_lo": 231.5,
 "entry_v_hi": 246.933
}
```
