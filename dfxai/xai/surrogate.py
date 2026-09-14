"""
설명가능성 지표: 결정트리 대리모델 충실도.

측정 방식
---------
정책이 실제로 남긴 (관측, 행동) 궤적에 결정트리 회귀모델을 적합시키고,
"얼마나 단순한 규칙으로 이 정책의 행동을 재현할 수 있는가"를 잽니다.

  - 충실도 F(d) : 깊이 d 트리가 원 정책의 행동을 재현한 정도 (R^2)
  - D*         : 목표 충실도(기본 0.90)에 도달하는 **최소 트리 깊이**
  - F(4)       : 깊이 4 고정 트리의 충실도 (보조 지표)

D* 가 작을수록 더 단순한 규칙으로 설명 가능 = 설명가능성이 높습니다.

왜 이 지표인가
--------------
SHAP·보상분해 같은 기존 설명기법은 대부분 특정 모델 계열에만 적용됩니다.
반면 이 지표는 **행동 로그만 있으면 계산**되므로 BT·RL·하이브리드에
완전히 동일하게 적용됩니다. 세 모델을 같은 잣대로 비교한다는 이 논문의
주장이 성립하는 근거가 바로 여기입니다.

"BT는 원래 트리라서 유리한 것 아니냐"는 반론에 대한 답
------------------------------------------------------
대리모델은 정책의 내부 구조가 아니라 **관측된 행동**에만 적합시킵니다.
게다가 이 프로젝트는 세 모델이 동일한 관측 벡터와 동일한 3차원 지령
공간을 공유하도록 설계했습니다. 따라서 자료형이 주는 이점은 없습니다.
실제로 BT-v2는 노드가 2개뿐이지만 arctan2 기반 연속 함수라
얕은 트리로는 잘 재현되지 않습니다.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, asdict
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

from ..config import FEATURE_NAMES, ACTION_NAMES

EPS_VAR = 1e-8


def _r2_per_output(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """출력 차원별 R^2. 분산이 0인 차원(항상 같은 값)은 1.0으로 처리합니다.

    BT-v2 처럼 스로틀을 항상 최대로 두는 정책은 해당 차원의 분산이 0인데,
    이는 '완벽히 설명 가능한 행동'이므로 R^2=1 이 옳습니다.
    sklearn 기본값은 이 경우 0 또는 nan을 주므로 직접 계산합니다.
    """
    out = np.empty(y_true.shape[1])
    for j in range(y_true.shape[1]):
        var = float(np.var(y_true[:, j]))
        if var < EPS_VAR:
            out[j] = 1.0
        else:
            sse = float(np.mean((y_true[:, j] - y_pred[:, j]) ** 2))
            out[j] = 1.0 - sse / var
    return out


@dataclass
class SurrogateResult:
    depths: list
    fidelity: list           # 깊이별 평균 R^2 (테스트셋)
    fidelity_per_action: list
    min_depth_at_target: float   # D* (도달 실패 시 np.inf 대신 max_depth+1)
    target: float
    fidelity_at_fixed_depth: float
    fixed_depth: int
    n_leaves_at_target: float
    n_samples: int
    d_star_multi: dict = None        # 여러 목표 충실도에서의 D* (강건성 확인용)
    action_var: list = None          # 채널별 분산 (퇴화 채널 확인용)

    def to_dict(self) -> dict:
        return asdict(self)


def surrogate_fidelity(obs: np.ndarray, act: np.ndarray,
                       depths=tuple(range(1, 17)), target: float = 0.95,
                       fixed_depth: int = 4, test_size: float = 0.3,
                       seed: int = 0, max_samples: int = 60_000) -> SurrogateResult:
    """(관측, 행동) 궤적으로부터 설명가능성 지표를 계산합니다."""
    obs = np.asarray(obs, dtype=np.float64)
    act = np.asarray(act, dtype=np.float64)
    assert obs.shape[0] == act.shape[0] and obs.shape[0] > 50, "샘플이 너무 적습니다"

    if obs.shape[0] > max_samples:                     # 과도한 로그는 균등 추출
        idx = np.linspace(0, obs.shape[0] - 1, max_samples).astype(int)
        obs, act = obs[idx], act[idx]

    Xtr, Xte, ytr, yte = train_test_split(obs, act, test_size=test_size,
                                          random_state=seed, shuffle=True)
    # 분산가중 평균: 항상 같은 값을 내는 채널(분산 0)은 가중치 0이 되어
    # 지표를 부풀리지 않습니다. 세 모델의 '움직이는 채널'만 공정하게 비교됩니다.
    var = np.var(yte, axis=0)
    w = var / var.sum() if var.sum() > EPS_VAR else np.ones(yte.shape[1]) / yte.shape[1]

    fids, fids_pa, leaves = [], [], []
    for d in depths:
        tree = DecisionTreeRegressor(max_depth=int(d), random_state=seed)
        tree.fit(Xtr, ytr)
        r2 = _r2_per_output(yte, tree.predict(Xte))
        fids.append(float(np.dot(w, r2)))
        fids_pa.append([float(v) for v in r2])
        leaves.append(int(tree.get_n_leaves()))

    fids_arr = np.asarray(fids)
    hit = np.where(fids_arr >= target)[0]
    if hit.size:
        d_star = float(depths[hit[0]])
        n_leaf = float(leaves[hit[0]])
    else:
        # 미달 시 '검사한 최대 깊이 + 1'로 우측 절단 처리하고, 논문에는
        # 절단 사실을 반드시 명시하세요(생존분석의 censoring과 같은 취급).
        d_star = float(max(depths) + 1)
        n_leaf = float(leaves[-1])

    # 목표값 하나에만 의존하지 않도록 여러 목표에서 D* 를 함께 보고합니다.
    # BT처럼 단순한 정책은 목표 0.90에서 전부 D*=1 로 뭉개지므로
    # 주 지표는 0.95, 강건성 확인용으로 0.90 / 0.99 를 함께 씁니다.
    d_star_multi = {}
    for tgt in (0.90, 0.95, 0.99):
        h = np.where(fids_arr >= tgt)[0]
        d_star_multi[str(tgt)] = float(depths[h[0]]) if h.size else float(max(depths) + 1)

    fixed_idx = list(depths).index(fixed_depth) if fixed_depth in depths else 0
    return SurrogateResult(
        depths=[int(d) for d in depths],
        fidelity=fids,
        fidelity_per_action=fids_pa,
        min_depth_at_target=d_star,
        target=target,
        fidelity_at_fixed_depth=float(fids[fixed_idx]),
        fixed_depth=int(fixed_depth),
        n_leaves_at_target=n_leaf,
        n_samples=int(obs.shape[0]),
        action_var=[float(v) for v in var],
        d_star_multi=d_star_multi,
    )


def rule_extract(obs: np.ndarray, act: np.ndarray, depth: int = 3,
                 seed: int = 0) -> str:
    """사람이 읽을 수 있는 규칙 텍스트를 뽑습니다.

    논문 본문에 '이 RL 정책을 깊이 3 규칙으로 근사하면 이렇게 읽힌다'는
    식으로 정성적 예시를 싣기 좋습니다. 정량 지표의 해석을 도와줍니다.
    """
    from sklearn.tree import export_text
    tree = DecisionTreeRegressor(max_depth=depth, random_state=seed)
    tree.fit(obs, act)
    return export_text(tree, feature_names=list(FEATURE_NAMES))
