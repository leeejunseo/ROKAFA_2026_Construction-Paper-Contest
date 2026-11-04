# BT-M+(교범 기반 행동트리) G1 검증 결과 (자동 생성)

노드 32개(안전 하위트리 7개), 자율 G 제한 7.7 G. 평가: 시드 10000–10099 × 진영 교대 × 상대 BT-v2·BT-v3 = 환경당 400교전.

## 1. G1 판정

| 기준 | 값 | 판정 |
|---|---|---|
| E1 3자유도 400교전 오류 없음 | 400/400 | 충족 |
| E2 3자유도 공중충돌 ≤ 10% | 0.020 | 충족 |
| E3 3자유도 지면충돌 ≤ 5% | 0.000 | 충족 |
| E4 3자유도 BT-v2 상대 점수 ≥ 0.45 | 0.432 [0.380, 0.482] | 미충족 |
| E1 6자유도 400교전 오류 없음 | 400/400 | 충족 |
| E2 6자유도 공중충돌 ≤ 10% | 0.035 | 충족 |
| E3 6자유도 지면충돌 ≤ 5% | 0.000 | 충족 |
| E4 6자유도 BT-v2 상대 점수 ≥ 0.45 | 0.532 [0.495, 0.570] | 충족 |
| E5 행동복제 R² ≥ 0.9 | 0.863 (채널별 0.888, 0.937, 0.724) | 미충족 |

## 2. 환경별 요약

| 지표 | 3자유도 | 6자유도 |
|---|---|---|
| 점수 vs BT-v2 | 0.4325 | 0.5325 |
| 점수 vs BT-v3 | 0.6325 | 0.5650 |
| 공중충돌률 | 0.0200 | 0.0350 |
| 자기 추락률 | 0.0000 | 0.0000 |
| 하드덱 위반율(스텝) | 0.0000 | 0.0000 |
| 과G 위반율(스텝) | 0.0000 | 0.0263 |
| 최소이격 위반율(스텝) | 0.0058 | 0.0018 |
| 6자유도 트림 성공률 | 1.0000 | 1.0000 |
| 실행 시간 [s] | 7.3999 | 166.1825 |

## 3. 결과 유형

- 3자유도 vs BT-v2: {'timeout:무': 88, 'gun_kill:패': 31, 'gun_kill:승': 4, 'timeout:패': 38, 'collision:무': 1, 'timeout:승': 38}
- 3자유도 vs BT-v3: {'collision:무': 7, 'timeout:무': 48, 'timeout:승': 96, 'gun_kill:승': 3, 'timeout:패': 21, 'gun_kill:패': 25}
- 6자유도 vs BT-v2: {'timeout:무': 131, 'timeout:승': 38, 'collision:무': 6, 'timeout:패': 23, 'gun_kill:패': 2}
- 6자유도 vs BT-v3: {'timeout:무': 112, 'collision:무': 8, 'timeout:패': 26, 'gun_kill:승': 3, 'timeout:승': 50, 'gun_kill:패': 1}

## 4. 설명가능성(참고)

| 환경 | D*(0.95) | D*(0.90/0.95/0.99) | F(4) | 샘플 |
|---|---|---|---|---|
| 3자유도 | 17 | {'0.9': 17.0, '0.95': 17.0, '0.99': 17.0} | 0.701 | 30714 |
| 6자유도 | 17 | {'0.9': 17.0, '0.95': 17.0, '0.99': 17.0} | 0.503 | 34529 |

행동복제 학습기: sklearn MLPRegressor(tanh 32-32) — torch 없음, 샘플 163,617개.

## 5. 노드 발동 빈도

| 노드 | 3자유도 | 6자유도 |
|---|---|---|
| rate_turn_lead_or_two_circle | 0.764 | 0.674 |
| high_g_pull | 0.097 | 0.041 |
| turn_circle_entry | 0.031 | 0.218 |
| break_turn_into_threat | 0.030 | 0.012 |
| relax_pull | 0.017 | 0.006 |
| recover_lift_vector_up | 0.016 | 0.033 |
| lead_pursuit_gun | 0.016 | 0.008 |
| rolling_tuck_under_jink | 0.013 | 0.001 |
| lateral_offset | 0.012 | 0.006 |
| lag_idle_reposition | 0.004 | 0.000 |

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
 "floor_lookahead": 5.0,
 "floor_margin": 300.0,
 "floor_zone": 1219.2,
 "v_rate_hi": 226.355,
 "thr_fast": 0.3,
 "speed_ref_alt": -1.0,
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
 "use_jink": true,
 "jink_ata": 15.0,
 "jink_range_scale": 1.1,
 "jink_alt": 609.6,
 "jink_roll": 90.0,
 "jink_relax_bank": 135.0,
 "jink_relax_g": 2.0,
 "use_range_control": true,
 "range_floor": 1219.2,
 "relax_g": 4.0,
 "use_overshoot_lag": true,
 "overshoot_range": 304.8,
 "lag_gain": 0.3,
 "use_closure_idle": true,
 "idle_closure": 30.0,
 "idle_range": 914.4,
 "use_lv_bias": true,
 "tc_lv_bias": 3.0,
 "rate_lv_bias": -3.0,
 "entry_climb_bias": 3.0,
 "use_entry_speed": true,
 "entry_v_lo": 231.5,
 "entry_v_hi": 246.933
}
```
