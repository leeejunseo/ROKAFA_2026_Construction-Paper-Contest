"""
행동트리 이름표 (C안).

학습·평가 스크립트는 행동트리를 이름으로 부른다. 기준선 BT 를 바꿔도(새 교범 BT → 교내대회 BT → BT-v2)
스크립트는 그대로 두고 이름만 바꾸면 되게 하려는 것이다.

    v1, v2, v3 : 기존 행동트리 3종 (agents/bt.py). 정수 1, 2, 3 도 같은 뜻
    btm        : 교범 기반 새 기준선 BT-M (손형래, agents/bt_manual.py — 형식 맞춤은 bt_manual_adapter.py)
    btmplus    : BT-M+ (상세판, 같은 파일)
    school     : 교내대회 BT 이식본 (박윤준, agents/bt_school.py 의 SchoolBTPolicy)

새 BT 가 지켜야 할 것 (C안_실행명세서 H1·H3, 예시는 agents/bt_template.py)
  * agents/base.py 의 Policy 를 상속하고 act(obs) 가 [-1, 1]^3 (뱅크, 하중배수, 스로틀) 을 낸다
  * 매 act() 뒤 self.last_node 에 마지막으로 실행한 행동 노드 이름을 남긴다
  * SAFETY_NODES : 감독형 혼합이 가져다 쓸 안전 행동 노드 이름들
  * PARAM_SPEC   : 파라미터 최적화형이 움직일 상수 [(이름, 하한, 상한, 기본값), …],
                   생성자 __init__(self, theta=None) — theta 는 [-1, 1]^k, None 이면 기본값
  * rule_count() : 논문에 보고할 규칙 수
"""
from __future__ import annotations
import importlib

from .base import Policy

# 이름 -> (모듈, 클래스). 파일이 아직 없으면 make_bt() 가 안내 메시지와 함께 실패한다.
_EXTERNAL = {
    "btm": ("dfxai.agents.bt_manual_adapter", "BTMPolicy"),
    "btmplus": ("dfxai.agents.bt_manual_adapter", "BTMPlusPolicy"),
    "school": ("dfxai.agents.bt_school", "SchoolBTPolicy"),
    "tmpl": ("dfxai.agents.bt_template", "TemplateBTPolicy"),     # 형식 예시·배선 시험용
}
_LABEL = {"btm": "BT-M", "btmplus": "BT-M+", "school": "BT-school", "tmpl": "BT-tmpl"}
_ALIAS = {"M": "btm", "M+": "btmplus"}           # 'BT-M' 'BT-M+' 같은 표 이름도 받는다


def canon(name) -> str:
    """1 / '1' / 'v1' / 'BT-v1' -> 'v1',  'BT-M' -> 'btm',  'BT-M+' -> 'btmplus'."""
    s = str(name).strip()
    if s.startswith("BT-"):
        s = s[3:]
    if s.isdigit():
        s = f"v{int(s)}"
    return _ALIAS.get(s, s)


def label(name) -> str:
    """결과 표에 쓰는 이름. 기존 결과와 같도록 v2 는 'BT-v2'."""
    c = canon(name)
    return _LABEL.get(c, f"BT-{c}")


def _external_class(c: str):
    mod, cls = _EXTERNAL[c]
    try:
        return getattr(importlib.import_module(mod), cls)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"행동트리 '{c}' 의 파일 {mod.replace('.', '/')}.py 가 아직 없습니다 "
            f"(C안: school=박윤준 Y2). 형식은 dfxai/agents/bt_template.py 참고") from e


def make_bt(name, theta=None) -> Policy:
    """이름으로 행동트리 정책을 만든다. theta 는 파라미터 최적화형에서만 준다."""
    c = canon(name)
    if c in ("v1", "v2", "v3"):
        if theta is not None:
            if c != "v3":
                raise ValueError(f"BT-{c} 는 조정 가능한 상수를 공개하지 않습니다 (v3 구조만 bt_param.py 로 가능)")
            from .bt_param import ParamBTPolicy
            return ParamBTPolicy(theta=theta)
        from .bt import BTPolicy
        return BTPolicy(version=int(c[1:]))
    if c in _EXTERNAL:
        cls = _external_class(c)
        return cls(theta=theta) if theta is not None else cls()
    raise ValueError(f"모르는 행동트리 이름: {name!r} (v1, v2, v3, btm, btmplus, school)")


def param_spec(name) -> list:
    """파라미터 최적화형이 움직일 상수 목록 [(이름, 하한, 상한, 기본값), …]."""
    c = canon(name)
    if c == "v3":
        from .bt_param import PARAM_SPEC
        return list(PARAM_SPEC)
    if c in _EXTERNAL:
        spec = getattr(_external_class(c), "PARAM_SPEC", None)
        if spec:
            return list(spec)
    raise ValueError(f"행동트리 '{c}' 는 PARAM_SPEC(조정 가능한 상수)을 공개하지 않습니다")


def default_theta(name):
    """PARAM_SPEC 기본값에 해당하는 θ ∈ [-1, 1]^k (파라미터 최적화형의 출발점)."""
    import numpy as np
    return np.asarray([2.0 * (d - lo) / (hi - lo) - 1.0 for _, lo, hi, d in param_spec(name)])


def safety_nodes(policy: Policy) -> tuple:
    """감독형 혼합이 가로챌 안전 노드. 정책(또는 그 모듈)이 SAFETY_NODES 를 안 밝히면 기존 BT-v3 의 두 노드."""
    import sys
    sn = getattr(policy, "SAFETY_NODES", None) or getattr(sys.modules.get(type(policy).__module__), "SAFETY_NODES", None)
    return tuple(sn or ("merge_offset", "level_climb"))
