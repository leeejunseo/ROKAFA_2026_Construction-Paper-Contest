# BT-M / BT-M+ 교범 대응표

작성: 손형래(H1 담당), 2026-11-04 초안. 코드: `dfxai/agents/bt_manual.py`, 검증: `dfxai/experiments/validate_btm.py`.

## 1. 근거 교범

| 약칭 | 문헌 | 비고 |
|---|---|---|
| KAF | Korean Air Force Tactics, Techniques & Procedures 3-3 Vol.5, *Basic Employment Manual F-16C*, 1 Oct 2005 | 표지·각 쪽 UNCLASSIFIED. 주 근거 |
| NOR | Royal Norwegian AF TTP 3-3 Vol.5, *Basic Employment Manual F-16AM*, 15 Apr 2001 | 같은 미 공군 AFTTP 3-3 계열. 같은 내용이 있는지 확인용 |

**보안 확인 필요.** 두 문헌 모두 비밀 표시는 없다. 하지만 일반 공개가 허가된 문서인지는 확인하지 못했다. 논문에 KAF를 인용하기 전에 지도교수님(보안성 검토)께 확인을 받는다. 인용이 안 되면 공개 출판물인 Shaw, *Fighter Combat: Tactics and Maneuvering*(1985)의 해당 장으로 출처 열만 바꾼다. 규칙 내용은 Shaw에도 있는 일반 BFM 원칙이라 코드는 바꿀 필요가 없다. 이 표와 원고에는 교범 원문을 옮기지 않고 절 번호와 한 줄 요약만 적는다.

## 2. 단위 변환

교범 값은 ft·KCAS다. 코드에는 m·m/s로 바꿔 넣었다.

- 1 ft = 0.3048 m, 1 kt = 0.5144 m/s.
- 속도 문턱값은 KCAS를 등가대기속도(EAS)로 본다. 저마하에서는 CAS ≈ EAS다.
- BT-M은 이 값을 기준고도 4,000 m(초기 고도대 3~5 km의 중앙)의 진대기속도로 한 번 환산해 자기 속도와 비교한다(`speed_ref_alt`). 예를 들어 440 KCAS → 226 m/s EAS → 약 277 m/s TAS다.
- BT-M+는 매 순간 그 고도의 EAS로 비교한다. 대리모델 재현성 비교는 4절에 있다.

## 3. 규칙 대응

열 설명:
- "판"은 BT-M(기준선)에 있는 규칙이면 M, BT-M+에만 있으면 M+다.
- NOR 열의 ○는 노르웨이판에서 같은 수치나 같은 기법을 찾았다는 뜻이다. "–"는 찾지 못했다는 뜻이다(본문 검색 기준).

### 3.1 안전 (SAFETY 하위트리, 감독형 혼합이 그대로 사용)

| 노드 | 판 | 규칙(코드) | 교범 근거 (KAF) | 교범 값 → 코드 값 | NOR |
|---|---|---|---|---|---|
| `pass_deconfliction` | M | 정면 조우에서 최근접점 예측 이격이 버블 안이고 2.5초 안이면, 상대가 지나갈 쪽의 반대로 75° 뱅크·6 G 회피 | 4.6.5.1 통과 시 최소 이격 버블 | 500 ft(연속훈련) → 152 m. 시뮬레이터 최소이격 규칙 150 m와 같은 크기 | ○ (bubble) |
| `floor_recovery` | M | 5초 뒤 예상 고도가 하드덱+300 m 아래면 날개 수평, 최대 G 당김, 최대 추력 | 4.6.32 Floor transitions(바닥 근처에서는 하강 기동 금지), 4.4.10 "3,500~4,000 ft above the floor" | 하드덱 1,000 m는 시뮬레이터 규칙에서 읽음 | ○ (floor transition) |
| (모든 추격 행동) | M | 하드덱 위 1,219 m 안에서는 기수 하향 선회(뱅크 > 90°)를 수평 선회로 제한 | 4.6.32 "less than 4,000 feet above … aggressive vertical down maneuver is no longer an option" | 4,000 ft → 1,219 m | – |
| (모든 행동) | M | 하중 지령 상한 7.7 G | 4.5.13 Maximum G pull "as body/stores limits allow" | 규칙 과G 한계 8 G − 여유 0.3 G | – |

### 3.2 방어 BFM (4.5)

| 노드 | 판 | 규칙 | 교범 근거 | 값 | NOR |
|---|---|---|---|---|---|
| `threatened` | M | 상대 기수가 나를 45° 안으로 겨누고, 상대가 내 3/9 선 뒤에 있으며, 거리가 2,743 m 안 | 4.3.3.6 WEZ(최대 ATA 45° 기준), 4.4.4 9,000 ft set, 4.5.5 생존 최우선 | 9,000 ft → 2,743 m | ○ |
| `break_turn_into_threat` | M | 양력벡터를 상대 쪽으로, 최대 G(7.7) | 4.5.10~4.5.13 Initial break turn | 미사일 대응용 "idle·양력벡터 약간 아래"는 제외(기총 전용 시뮬레이터) | ○ (break turn) |
| `rolling_tuck_under_jink` | M+ | 상대 기총 WEZ 안(상대 ATA < 15°, 거리 < 1,320 m)이고 고도 여유가 있으면, 선회 방향으로 계속 굴리며 135° 넘으면 당김 완화 | 4.5.35~4.5.36 Low-aspect gun jinks, rolling tuck under | 1,500~2,000 ft above floor → 610 m, "approximately 135 degrees" | ○ (jink 기술은 있음, tuck-under 명칭은 –) |

### 3.3 공격 BFM (4.4)

| 노드 | 판 | 규칙 | 교범 근거 | 값 | NOR |
|---|---|---|---|---|---|
| `offensive_position` | M | 내가 상대 후방 반구(AA < 90°)이고 기수가 상대 쪽(ATA < 90°) | 4.3.2~4.3.3 위치 기하(거리·AA·HCA·ATA) | | ○ |
| `lead_pursuit_gun` | M | 거리 1,372 m 안, 상대가 기수 40° 안 → 리드추격(이득 1.3), 최대 G | 4.4.16 Low-aspect gunshot "range 3,500 to 4,500 feet … transition to lead", "30 to 40 degrees out the top of the HUD" | 4,500 ft → 1,372 m | – (수치) |
| `high_g_pull` | M | 선회원(1,829 m) 안 → 순수추격 최대 G | 4.4.4 반경 약 3,000 ft → 선회원 6,000 ft, 4.4.5.2 "high-G pull (typically 6 G to 8 G)" | 6,000 ft → 1,829 m | ○ |
| `turn_circle_entry` | M | 선회원 밖 → 상대 쪽으로 4 G 접근(에너지 보존) | 4.4.4.1~4.4.4.3 Turn-circle entry, "least-moving parts" | | ○ |
| `relax_pull` | M+ | 선회원 안에서 1,219 m보다 가까워지고 접근 중이면 4 G로 완화 | 4.4.5.3 "If the slant range decreases to less than 4,000 feet … relax the pull" | 4,000 ft → 1,219 m | – |
| `lag_idle_reposition` | M+ | 305 m 안에서 접근 중이면 래그추격(이득 0.3)·아이들 | 4.4.16 "shoot-lag-shoot", 4.4.17 Reposition | 1,000 ft → 305 m | ○ (lag) |
| (`lead_pursuit_gun` 안) | M+ | 리드 확보 후 접근률 > 30 m/s, 거리 < 914 m면 아이들 | 4.4.16 "Pull power to idle and extend speed brakes once lead is assured" | | – |
| (`high_g_pull` 안) | M+ | 양력벡터를 상대 POM보다 약 3° 위 | 4.4.5.2 "place the gun cross just above the adversary's POM" | | – |
| (`turn_circle_entry` 안) | M+ | 진입 속도 450~480 KCAS 유지, 넘으면 2~3° 상승 | 4.4.4.2 "Accelerate level in AB until 450 KCAS to 480 KCAS", "slight climb (2 to 3 degrees)" | 450/480 KCAS → 232/247 m/s EAS | ○ (450~480 KCAS) |

### 3.4 정면 BFM과 에너지 관리 (4.6, 4.3.3)

| 노드 | 판 | 규칙 | 교범 근거 | 값 | NOR |
|---|---|---|---|---|---|
| `rate_turn_lead_or_two_circle` | M | 위 조건이 모두 아니면 상대 쪽으로 최대 G 선회. 통과 전에는 리드 턴, 통과 후에는 2서클 선회율 싸움 | 4.6.5.3 Lead turn, 4.6.9.1 Two-circle(선회율 우위), 4.6.40 Equal adversary | | ○ (lead turn, two-circle) |
| 스로틀(모든 행동) | M | 등가속도 440 KCAS를 넘으면 스로틀 0.3, 아니면 최대 | 4.3.3.7 "corner plateau … between 330 to 440 KCAS", 4.6.12 Rate versus radius | 440 KCAS → 226 m/s EAS | ○ (corner plateau) |
| (`rate_turn` 안) | M+ | 바닥 구역 밖에서는 양력벡터를 약 3° 아래(nose-low pull) | 4.6.18.2 "a nose-low pull produces the greatest rate with the lowest airspeed bleed" | | – |

### 3.5 옮기지 않은 교범 내용과 이유

| 교범 내용 | 이유 |
|---|---|
| 미사일 사격·회피, IRCM(채프/플레어), idle break(IR 미사일 대응) | 시뮬레이터는 기총만 있음 |
| 1서클(반경) 싸움 선택(4.6.9.2) | 상대의 선회 방향을 관측 20개로 알 수 없음. 상태 없는 규칙(통과 후 선회 방향 유지)으로 만들어 시험한 결과, BT-v2 상대 점수가 켰을 때 0.36~0.43, 껐을 때 0.47~0.50(시험 200~300교전) |
| 수직 기동 세부(4.4.10 pure vertical, 4.6.31) | 3자유도 점질량 모델의 경로각 제한(±82°)과 관측 범위 밖. 하강 금지 구역 규칙만 반영 |
| 시각 단서(캐노피 위 LOS 이동, "rotation vs translation") | 관측 벡터에 LOS 변화율이 없음. 거리·각도 문턱값으로 대신함 |
| 상대 기종별 게임플랜(4.6.33~4.6.39) | 같은 기체끼리의 교전만 있음 |

## 4. 두 판의 검증 결과 요약 (G1)

평가 방법: 시드 10000~10099 × 진영 교대 × 상대 BT-v2·BT-v3로 환경마다 400교전을 치렀다. 자세한 값은 `results/btm_g1/BT-M/G1_report.md`와 `results/btm_g1/BT-M+/G1_report.md`에 있다.

| | BT-M (22노드) | BT-M+ (32노드) | 기준 |
|---|---|---|---|
| 3자유도 점수 vs BT-v2 / v3 | 0.473 / 0.673 | 0.433 / 0.633 | v2 ≥ 0.45 |
| 6자유도 점수 vs BT-v2 / v3 | 0.465 / 0.525 | 0.533 / 0.565 | v2 ≥ 0.45 |
| 공중충돌 (3 / 6자유도) | 0.010 / 0.048 | 0.020 / 0.035 | ≤ 0.10 |
| 자기 추락 | 0 / 0 | 0 / 0 | ≤ 0.05 |
| 하드덱 위반 | 0 / 0 | 0 / 0 | (참고) |
| 과G 위반(스텝, 6자유도) | 0.025 | 0.026 | (참고) |
| D\*(0.95) [D\*(0.90)] | 7 [4] (3·6자유도 같음) | 17 [17] (절단) | (참고) |
| 행동복제 R² (sklearn 대용, 3자유도) | 0.919 (채널별 0.966 / 0.976 / 0.714) | 0.863 (반복 300회 설정, 수렴 전) | ≥ 0.9 |

판단은 이렇다.
- 기준선으로는 BT-M을 쓴다. G1의 E1~E5를 모두 충족한다. 다만 6자유도 BT-v2 상대 점수 0.465는 95% 구간 [0.430, 0.500]의 하한이 0.45 아래라서 "기준을 넉넉히 넘었다"고 쓰지 않는다. 행동복제 R²는 torch가 없는 환경에서 같은 구조의 sklearn MLP로 잰 값이다. 본 실험의 `pretrain_bc.fit`(torch)으로 한 번 더 확인해야 한다. D\*가 7로 기존 BT-v3(5)와 비슷해, 하이브리드의 "규칙 쪽 기준점" 역할을 한다.
- BT-M+는 교범을 더 자세히 옮긴 판이다. 6자유도 성능은 더 좋지만 D\*가 17로 절단된다. 드물게 발동하는 규칙(저크·오버슛·아이들)이 많을수록 결정트리 재현이 어려워지기 때문이다. 기준선으로는 쓰지 않는다. 다만 "교범을 충실히 옮긴 규칙 모델도 D\*로는 복잡하게 측정된다"는 보조 근거로 쓸 수 있다.
- 6자유도 과G 위반 약 2.5%는 G 상한을 6.5 G로 낮춰도 0.6% 남는다(시험 100교전). 정책보다는 추종 제어기의 하중 오버슛이나 Nz 측정(기존 E3 문제)일 가능성이 크다. 이준서 조원 확인이 필요하다.

## 5. 설계 중 바꾼 것 (기록)

1. **통과 회피 범위.** 처음에는 정면 조우 판정을 좁게 잡았더니 공중충돌이 8%였다. 버블 1,000 ft로 넓히자 회피가 너무 잦아 BT-v2 상대 점수가 0.18~0.30으로 떨어졌다. 최종안은 최근접점 예측 + 버블 500 ft(교범 연속훈련 기준)다.
2. **1서클 규칙 제외.** 3.5절 참고.
3. **양력벡터 편향의 불연속 수정.** 상대가 기수 정면에 있을 때 편향만 남으면 배면 뱅크 지령이 나왔다. 수평 이탈각 30°까지 비례해서 거는 방식으로 고쳤다.
4. **에너지 관리 규칙.** 3자유도에서는 점수 차이가 거의 없었다. 6자유도에서 빼면 BT-v2 상대 점수가 0.47에서 0.19로 무너졌다(시험 100교전). 그래서 기준선에 남겼다. 교범의 속도 관리가 실제 F-16 비행 모델에서 효과를 낸다는 근거로 원고에 쓸 수 있다.
5. **속도 판단 기준.** 매 순간 EAS로 판단하면 결정트리가 (속도, 고도) 곡선 경계를 계단으로 근사해야 해서 D\*가 17로 절단됐다. 기준고도 4 km의 진대기속도로 고정 환산하자 D\* 7이 됐고 성능은 비슷했다. BT-M은 고정 환산을, BT-M+는 EAS를 쓴다.

## 6. 학습·평가 상대 BT-E, BT-R (2026-11-05 추가)

BT-v1·v2·v3 대신, 같은 교범의 서로 다른 게임플랜으로 학습·평가 상대를 새로 만들었다.

| 모델 | 역할 | 게임플랜 | 노드 | 코드 |
|---|---|---|---|---|
| BT-M | 기준선, 학습 상대 | 선회율 싸움: 코너 플래토(330~440 KCAS)에서 최대 G 2서클 | 22 | `bt_manual.py` |
| BT-E | 학습 상대 | 에너지·수직 싸움: 선회원 밖에서는 지속 선회 하중으로 에너지를 아끼고, 기수 하향 면외 기동과 가속 이탈을 쓴다 | 25 | `bt_energy.py` |
| BT-R | 평가 전용 | 반경 싸움: 330 KCAS 위에서는 추력을 줄여 빨리 감속하고, 아래에서 최대 G 로 가장 작은 반경으로 기수를 먼저 돌린다. 정면에 가까운 각도에서도 사격한다 | 15 | `bt_radius.py` |
| BT-M+ | 평가 전용 | BT-M 에 교범 세부 기법(저크, 오버슛 래그, 아이들, 양력벡터 편향, 진입 속도창)을 더한 상세판 | 32 | `bt_manual.py` |

네 모델은 안전 하위트리(통과 이격, 하드덱 회복), 바닥 구역의 하강 선회 제한, 7.7 G 상한을 공유한다. 감독형 하이브리드가 가로채는 안전 노드도 같다.

### 6.1 BT-E 규칙 대응

| 노드 | 규칙 | 교범 근거 (KAF) | 값 |
|---|---|---|---|
| `snapshot_lead` | 거리 1,372 m 안, 기수 40° 안이면 최대 G 리드 추격 | 4.4.16 | 4,500 ft, 40° |
| `break_turn_into_threat` | 위협받고 거리 1,829 m 안이거나 상대가 접근 중이면 브레이크 턴 | 4.5.10~4.5.13 | 6,000 ft |
| `unload_extend` | 위협받지만 이미 멀어지는 중이면 날개 수평, 0.8 G, 최대 추력으로 이탈 | 4.3.3.17 "0.7 G to 0.9 G slightly nose-low extension", 4.3.3.18 "a separation will usually only be successful if the adversary chooses not to pursue" | 0.8 G |
| `unload_accelerate` | 선회원 밖에서 330 KCAS 아래로 떨어지면 1.5 G 로 상대 쪽을 향하며 가속 | 4.6.12 Rate versus Radius, 4.3.3.18 Acceleration | 330 KCAS → 170 m/s EAS |
| `nose_low_lead` | 하드덱 위 1,524 m 이상, 상대와 914 m 이상 떨어져 있고 중립 이상이면, 양력벡터를 상대 리드점보다 10° 아래로 두고 최대 G | 4.6.31.1 "aggressive nose-low lift vector on to in lead maneuver", 4.3.3.14 "maximum G ... best employed when working with gravity", 4.6.32 4,000 ft 규칙 | 5,000 ft, 3,000 ft, −10° |
| `sustained_pursuit` | 그 밖: 선회원(1,829 m) 안에서는 최대 G, 밖에서는 지속 선회 하중 6.4 G. 440 KCAS 위에서만 추력 감소 | 표 4.9 정상 선회 425 KIAS 13.5°/s(약 6.4 G), 4.6.36.2 "keep your power up anytime less than 400 KCAS" | 6.4 G |

### 6.2 BT-R 규칙 대응

| 노드 | 규칙 | 교범 근거 (KAF) | 값 |
|---|---|---|---|
| (모든 행동) 추력 | 330 KCAS 위: 스로틀 0으로 빨리 감속. 아래: 최대 추력 | 4.6.36.1 "if it is above that, you will want to reduce power to minimize the radius as quickly as possible. As soon as you approach 330 KCAS, reselect AB" | 330 KCAS |
| `high_aspect_lead` | 거리 1,372 m 안, 기수 40° 안이면 애스펙트와 상관없이 리드 사격 | 4.4.15 High-Aspect Gunshot, 4.4.16 | 4,500 ft, 40° |
| `break_turn_into_threat` | 위협받으면 최대 G 브레이크 턴 | 4.5.10~4.5.13 | |
| `min_radius_turn` | 그 밖: 상대 쪽으로 최대 G (반경 구간에서 기수를 먼저 돌린다) | 4.6.9.2 one-circle "the deciding advantage … will be based on turn radius", 4.6.12 "130~329 KCAS gives us the tightest turn radius", 4.6.13 | |

### 6.3 공통 안전 하위트리 수정 (BT-M 포함)

6자유도 상호 검증에서 정면 통과 때 공중충돌이 기준을 넘었다(BT-R 대 BT-M+ 13.5%, BT-M 대 BT-M+ 10.0%). 기록을 보면 두 기체 모두 회피 노드(`lateral_offset`) 안에 있었고, 배면에 가까운 자세에서 바로 6 G 로 당겨 기수가 상대 고도 쪽으로 떨어졌다.
그래서 교범 4.4.5.2 "use an unloaded roll to set Lv"에 따라, 실제 뱅크가 회피 뱅크에서 30° 안에 들어올 때까지는 1 G 로 굴리기만 하도록 고쳤다(`pass_roll_tol`). 그 결과 공중충돌은 두 짝 모두 7.5%로 내려갔다.
같은 하위트리를 쓰는 BT-M 도 바뀌었으므로 G1 을 다시 검증하였다(`results/btm_g1/BT-M_v2/`). BT-v2 상대 점수는 3DOF 0.478, 6DOF 0.463 으로 수정 전(0.472, 0.465)과 거의 같다.

### 6.4 설계 중 바꾼 것 (BT-E)

모두 BT-M 상대 시험(각 60교전)으로 확인했다.
1. **지속 선회 하중을 5.5 G → 6.4 G 로 바꿨다.** 처음에는 표 4.9 를 잘못 읽어 5.5 G 로 두었다. 다시 읽은 결과 425 KCAS 행에서 6 G 는 +3 kt/s, 7 G 는 −5 kt/s 였고, 정상 선회(0 Ps) 수치 13.5°/s 는 약 6.4 G 에 해당했다. 5.5 G 일 때 6DOF BT-M 상대 점수는 0.13~0.26 으로 너무 약했다.
2. **가속은 선회원 밖에서만 하도록 했다.** 교전 중에 하중을 풀면 BT-M 에게 사격 기회를 줘서 격추가 42%까지 늘었다. 교범의 가속·이탈도 교전에서 벗어날 때의 기법이다.
3. **이탈은 이미 멀어지는 중일 때만 하도록 했다.** 4.3.3.18 이 "상대가 쫓지 않을 때만 성공한다"고 한 것과 같다.
4. **기수 하향 면외 기동은 양력벡터를 −25° 에서 −10° 로, 하중은 6 G 에서 최대 G 로 바꿨다.** 교범은 이 기동의 각도 수치를 주지 않는다. −25° 일 때 BT-M 상대 점수가 3DOF 0.21, 6DOF 0.13 이어서 시험으로 −10° 를 골랐다. 하중은 4.3.3.14 대로 중력 방향 선회이므로 최대 G 로 했다.

### 6.5 상호 검증 결과 (`results/opponents/report.md`)

짝마다 시드 10000~10099 × 진영 교대 = 400교전이다. 새 상대 BT-E, BT-R 은 3·6DOF 모두 판정 기준 O1~O4 를 충족했다(오류 0, 공중충돌 ≤ 10%, 추락 0, BT-M 상대 점수 0.35~0.65).

| 짝 (왼쪽 점수) | 3DOF | 6DOF |
|---|---|---|
| BT-M vs BT-E | 0.585 | 0.647 |
| BT-M vs BT-R | 0.450 | 0.487 |
| BT-M vs BT-M+ | 0.500 | 0.405 |
| BT-E vs BT-R | 0.427 | 0.505 |
| BT-E vs BT-M+ | 0.273 | 0.130 |
| BT-R vs BT-M+ | 0.545 | 0.268 |

해석할 때 주의할 점:
- **BT-E 는 6DOF 에서 BT-M 보다 약하다.** BT-M 상대 점수가 0.353 으로 기준 하한(0.35)에 가깝고, 95% 구간 하한은 그 아래다. 학습 상대로서 "너무 약하지 않은가"는 G2 시범 학습에서 다시 본다.
- **BT-M+ 는 6DOF 에서 가장 강하다.** BT-E 0.870, BT-R 0.732, BT-M 0.595 로 이긴다. 교범 세부 기법이 실제 F-16 비행 모델에서 효과를 낸다는 뜻이다. 평가 전용 상대로서는 어려운 시험 역할을 한다.
- **행동 차이:** BT-R 은 평균 등가속도가 6DOF 182 m/s 로 BT-M(239)보다 확실히 느리다. BT-E 는 속도 차이(243)보다 하중(6.67 G 대 6.86 G)과 고도(2,420 m 대 2,544 m) 차이로 구분된다.
