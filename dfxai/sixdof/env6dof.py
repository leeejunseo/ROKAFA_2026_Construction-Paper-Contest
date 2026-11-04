"""
6자유도 교전 환경: DogfightEnv 와 같은 교전 규칙·관측·지령 변환에, 비행역학만 JSBSim F-16.

바뀌는 것   : 두 기체의 운동 (DogfightEnv._integrate -> JSBAircraft.step)
그대로인 것 : 초기조건 생성(같은 시드 = 같은 위치·고도·속도·방위), action_to_command(),
             build_obs() 20개 관측, WEZ·체력·종료조건·120초·안전규칙 집계, 시간종료 판정

과G 위반 판정 (C안 E3, 2026-11-04): 3자유도의 기체 하중배수는 지령을 1차 지연으로 따라가므로 지령보다
위로 넘치지 않는다. 6자유도 추종 제어기는 지령 주위에서 오르내리므로, 8 G 를 정확히 지령하는 정책
(BT-v3, Shield)도 순간 하중이 8 G 를 살짝 넘는 적분 스텝이 40% 안팎 생겼다(중앙값 +0.07 G).
그래서 6자유도에서는 "기체가 실제로 8 G 를 넘었고(0.05초 평균 Nz) 정책도 8 G 넘게 당기라고 했다
(3자유도와 같은 1차 지연을 거친 지령)"일 때만 과G 로 센다. 3자유도에서는 둘이 같은 값이라 정의가 같다.

초기조건: DogfightEnv.reset() 이 만든 상태 그대로 두 기체를 수평 정상비행으로 트림해 띄운다.
지령 변환의 가용 하중배수(max_load_factor)는 3자유도 공식 그대로라서, 정책은 3자유도에서와
같은 의미의 지령을 낸다. 기체가 그 지령을 못 따라가면(롤 속도, 받음각·하중 한계) 그것이
곧 이 교차검증이 재려는 비행 모델 충실도의 차이다.
"""
from __future__ import annotations

from ..dynamics import max_load_factor
from ..env import DogfightEnv
from .aircraft import JSBAircraft


class SixDofEnv(DogfightEnv):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._jsb = (JSBAircraft(), JSBAircraft())      # 모델 로드는 한 번만, 교전마다 재초기화
        self._map: dict = {}
        self._integrate = self._jsb_integrate

    def reset(self, seed: int = 0, alpha: float = 1.0, alpha_red: float = 0.0):
        super().reset(seed=seed, alpha=alpha, alpha_red=alpha_red)
        for sim, st in zip(self._jsb, (self.blue, self.red)):
            sim.reset_from(st)
            sim.sync(st)
        self._map = {id(self.blue): self._jsb[0], id(self.red): self._jsb[1]}
        self._nlag = {id(self.blue): 1.0, id(self.red): 1.0}
        return self.obs_blue(), self.obs_red()

    def _jsb_integrate(self, state, cmd, dt, ac):
        sim = self._map[id(state)]
        # 3자유도 dynamics.step 과 같은 지령 포화·1차 지연 (과G 판정 전용, 비행에는 안 씀)
        k = id(state)
        n_c = min(max(float(cmd[1]), ac.n_min_struct), max_load_factor(state.v, state.h, ac))
        self._nlag[k] += (n_c - self._nlag[k]) * min(1.0, dt / ac.tau_load)
        sim.step(cmd, dt)
        sim.sync(state)

    def _rule_load(self, state) -> float:
        return min(self._nlag[id(state)], self._map[id(state)].nz_mean)

    @property
    def trim_ok(self) -> bool:
        return all(s.trim_ok for s in self._jsb)
