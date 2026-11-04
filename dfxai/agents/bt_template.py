"""
새 기준선 BT 가 지켜야 할 형식의 예시 (C안 H1·H3). 실험에는 쓰지 않는다.

새 BT 는 이 형식을 갖추면 registry 에 한 줄 올려 이름으로 부를 수 있다. 손형래의 BT-M(agents/bt_manual.py)은
agents/bt_manual_adapter.py 가 이 형식을 덧붙여 이름 btm 으로 불린다(agents/registry.py).
여기서는 BT-v3 와 같은 구조(bt_param.py)를 빌려 형식만 보인다. 이름표에는 'tmpl' 로 올라 있어
배선 시험(파라미터 최적화형 포함 4종 학습이 끝까지 도는지)에 쓴다.

형식 점검:  python -m dfxai.cplan.accept_bt --bt tmpl --quick
"""
from __future__ import annotations
import numpy as np

from .bt_param import ParamBTPolicy, PARAM_SPEC as _V3_SPEC


class TemplateBTPolicy(ParamBTPolicy):
    # (1) 파라미터 최적화형이 움직일 상수: (이름, 하한, 상한, 기본값). 5–10개 (H3)
    PARAM_SPEC = list(_V3_SPEC)
    # (2) 감독형 혼합이 가로챌 안전 행동 노드 이름 (act() 가 last_node 에 남기는 이름과 같아야 함)
    SAFETY_NODES = ("merge_offset", "level_climb")

    # (3) 생성자: theta ∈ [-1, 1]^k 또는 None(기본값)
    def __init__(self, theta: np.ndarray | None = None):
        super().__init__(theta=theta)
        self.name = "BT-tmpl"

    # (4) act(obs) -> np.array([뱅크, 하중배수, 스로틀]) ∈ [-1, 1]^3, 그리고 self.last_node 갱신
    #     (ParamBTPolicy.act 를 그대로 씀)

    # (5) 논문에 보고할 규칙 수 (노드 수 대신, H5)
    def rule_count(self) -> int:
        return 4
