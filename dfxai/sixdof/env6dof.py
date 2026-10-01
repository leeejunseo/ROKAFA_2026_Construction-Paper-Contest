"""
6자유도 교전 환경: DogfightEnv 와 같은 교전 규칙·관측·지령 변환에, 비행역학만 JSBSim F-16.

바뀌는 것   : 두 기체의 운동 (DogfightEnv._integrate -> JSBAircraft.step)
그대로인 것 : 초기조건 생성(같은 시드 = 같은 위치·고도·속도·방위), action_to_command(),
             build_obs() 20개 관측, WEZ·체력·종료조건·120초·안전규칙 집계, 시간종료 판정

초기조건: DogfightEnv.reset() 이 만든 상태 그대로 두 기체를 수평 정상비행으로 트림해 띄운다.
지령 변환의 가용 하중배수(max_load_factor)는 3자유도 공식 그대로라서, 정책은 3자유도에서와
같은 의미의 지령을 낸다. 기체가 그 지령을 못 따라가면(롤 속도, 받음각·하중 한계) 그것이
곧 이 교차검증이 재려는 비행 모델 충실도의 차이다.
"""
from __future__ import annotations

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
        return self.obs_blue(), self.obs_red()

    def _jsb_integrate(self, state, cmd, dt, ac):
        sim = self._map[id(state)]
        sim.step(cmd, dt)
        sim.sync(state)

    @property
    def trim_ok(self) -> bool:
        return all(s.trim_ok for s in self._jsb)
