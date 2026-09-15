"""
Gymnasium 래퍼 + Stable-Baselines3 PPO 학습.

ES 경로(train_es.py)와 이 PPO 경로는 **같은 MLPPolicy 포맷**으로
체크포인트를 저장하므로, 평가·분석 파이프라인을 그대로 공유합니다.
둘 중 먼저 결과가 나오는 쪽을 논문 본 실험으로 쓰고, 다른 쪽은
부록의 강건성 확인으로 돌리면 됩니다.

설치:
    pip install gymnasium stable-baselines3 torch

핵심 설계
---------
* 관측에 alpha 를 포함하고 에피소드마다 무작위 추출 -> 정책 하나로 전 구간 커버
* 학습 초반에는 alpha 하한을 낮춰 BT 가 탐색을 이끔 (자동 커리큘럼)
* 포텐셜 기반 셰이핑으로 희소보상 문제 해결 (최적정책 불변 보장)
* SubprocVecEnv 로 코어 수만큼 병렬화 — JSBSim이 아니라 numpy라 매우 빠름
* CheckpointCallback 으로 학습 도중 여러 지점을 저장 -> '학습 예산' 축
"""
from __future__ import annotations
import argparse, os
import numpy as np

from ..config import OBS_DIM, ACTION_DIM
from ..env import DogfightEnv
from ..agents.bt import BTPolicy
from ..agents.hybrid import MLPPolicy
from .shaping import potential, terminal_reward

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAS_GYM = True
except ImportError:                                    # 래퍼만 없을 뿐 나머지는 동작
    gym, spaces, _HAS_GYM = None, None, False


if _HAS_GYM:
    class DogfightGymEnv(gym.Env):
        """단일 에이전트 관점의 Gymnasium 환경.

        청군이 학습 대상, 홍군은 고정 BT 상대입니다.
        행동은 RL 성분 a_RL 이고, 환경 내부에서 BT 와 혼합됩니다.
        """
        metadata = {"render_modes": []}

        def __init__(self, opponents=(1, 2), bt_version: int = 2,
                     alpha_lo: float = 0.15, alpha_hi: float = 1.0,
                     gamma: float = 0.997, seed: int = 0):
            super().__init__()
            self.observation_space = spaces.Box(-5.0, 5.0, (OBS_DIM,), np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, (ACTION_DIM,), np.float32)
            self.env = DogfightEnv()
            self.bt = BTPolicy(version=bt_version)
            self.opponents = tuple(opponents)
            self.alpha_lo, self.alpha_hi = alpha_lo, alpha_hi
            self.gamma = gamma
            self.rng = np.random.default_rng(seed)

        def set_alpha_range(self, lo: float, hi: float = 1.0):
            """커리큘럼 콜백이 학습 진행에 따라 호출합니다."""
            self.alpha_lo, self.alpha_hi = float(lo), float(hi)

        def reset(self, *, seed=None, options=None):
            if seed is not None:
                self.rng = np.random.default_rng(seed)
            self.alpha = float(self.rng.uniform(self.alpha_lo, self.alpha_hi))
            self.opp = BTPolicy(version=int(self.rng.choice(self.opponents)))
            s = int(self.rng.integers(0, 10 ** 8))
            self.ob, self.orr = self.env.reset(seed=s, alpha=self.alpha)
            self.prev_pot = potential(self.ob)
            self.prev_dealt = self.prev_taken = 0.0
            return self.ob.astype(np.float32), {}

        def step(self, action):
            a_rl = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
            a_bt = self.bt.act(self.ob)
            a_blue = np.clip((1.0 - self.alpha) * a_bt + self.alpha * a_rl, -1.0, 1.0)
            a_red = self.opp.act(self.orr)

            self.ob, self.orr, done = self.env.step(a_blue, a_red)
            pot = potential(self.ob)
            dealt = self.env._dmg_dealt - self.prev_dealt
            taken = self.env._dmg_taken - self.prev_taken
            self.prev_dealt, self.prev_taken = self.env._dmg_dealt, self.env._dmg_taken

            r = 0.02 * dealt - 0.02 * taken
            r += self.gamma * pot - self.prev_pot        # 포텐셜 기반 셰이핑
            self.prev_pot = pot

            if done:
                res = self.env.result()
                r += terminal_reward(res.winner, res.outcome)
                # ES 적합도와 같은 취지: 양측 무피해 시간종료(회피 무승부) 벌점,
                # 규칙 위반 약한 벌점. 스텝 보상 스케일(피해 1점 = 0.02)에 맞춰 축소.
                if res.outcome == "timeout" and res.damage_dealt == 0.0 and res.damage_taken == 0.0:
                    r += -8.0
                v = res.violations_blue
                r -= 3.0 * (v.get("deck", 0.0) + v.get("over_g", 0.0))
            return self.ob.astype(np.float32), float(r), bool(done), False, {}


def sb3_to_mlp(model, out_path: str, hidden=(32, 32)) -> str:
    """SB3 정책의 가중치를 MLPPolicy 포맷으로 변환해 저장.

    SB3 기본 MlpPolicy 는 pi 네트워크(mlp_extractor.policy_net) + action_net 구조.
    net_arch 를 hidden 과 동일하게 두고 학습해야 형상이 맞습니다.
    """
    import torch
    pi = model.policy
    layers = []
    for m in pi.mlp_extractor.policy_net:
        if isinstance(m, torch.nn.Linear):
            layers.append(m)
    layers.append(pi.action_net)

    flat = []
    for lin in layers:
        W = lin.weight.detach().cpu().numpy().T      # (in, out) 으로 전치
        b = lin.bias.detach().cpu().numpy()
        flat.append(W.ravel()); flat.append(b.ravel())
    flat = np.concatenate(flat)

    p = MLPPolicy(hidden=hidden, params=flat, out_act="clip")
    p.save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="SB3 PPO 학습")
    ap.add_argument("--timesteps", type=int, default=3_000_000)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bt-version", type=int, default=2)
    ap.add_argument("--outdir", default="results/ppo")
    ap.add_argument("--save-every", type=int, default=250_000)
    ap.add_argument("--alpha-lo", type=float, default=0.15,
                    help="alpha 하한 (1.0 이면 BT 보조 없이 순수 학습 = 밑바닥 학습)")
    ap.add_argument("--alpha-hi", type=float, default=1.0)
    ap.add_argument("--no-curriculum", action="store_true",
                    help="alpha 커리큘럼 끄기 (alpha 범위를 고정)")
    ap.add_argument("--opponents", nargs="*", type=int, default=[1, 2])
    a = ap.parse_args()

    if not _HAS_GYM:
        raise SystemExit("gymnasium 이 없습니다. "
                         "pip install gymnasium stable-baselines3 torch "
                         "또는 ES 경로(train_es.py)를 쓰세요.")
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.callbacks import BaseCallback

    os.makedirs(a.outdir, exist_ok=True)
    hidden = (32, 32)

    def mk(i):
        def _f():
            return DogfightGymEnv(opponents=tuple(a.opponents), bt_version=a.bt_version,
                                  alpha_lo=a.alpha_lo, alpha_hi=a.alpha_hi,
                                  seed=a.seed * 1000 + i)
        return _f

    venv = SubprocVecEnv([mk(i) for i in range(a.n_envs)])
    model = PPO("MlpPolicy", venv, seed=a.seed, verbose=1,
                n_steps=512, batch_size=2048, gae_lambda=0.95, gamma=0.997,
                ent_coef=0.004, learning_rate=3e-4, clip_range=0.2,
                policy_kwargs=dict(net_arch=dict(pi=list(hidden), vf=[64, 64])))

    class Curriculum(BaseCallback):
        """학습 진행에 따라 alpha 하한을 끌어올려 BT 의존도를 점차 낮춥니다."""
        def _on_step(self) -> bool:
            if a.no_curriculum:
                return True
            if self.n_calls % 5000 == 0:
                frac = min(1.0, self.num_timesteps / (0.7 * a.timesteps))
                lo = a.alpha_lo + (0.9 - a.alpha_lo) * frac
                self.training_env.env_method("set_alpha_range", lo, a.alpha_hi)
            return True

    class Budget(BaseCallback):
        """학습 예산 축을 만드는 체크포인트 저장."""
        def _on_step(self) -> bool:
            if self.num_timesteps % a.save_every < a.n_envs:
                p = os.path.join(a.outdir,
                                 f"ckpt_seed{a.seed}_step{self.num_timesteps:09d}.npz")
                sb3_to_mlp(self.model, p, hidden)
            return True

    log = open(os.path.join(a.outdir, f"train_seed{a.seed}.log"), "a", encoding="utf-8")
    log.write(f"PPO seed={a.seed} timesteps={a.timesteps} alpha=[{a.alpha_lo},{a.alpha_hi}] "
              f"curriculum={not a.no_curriculum} opponents={a.opponents}" + chr(10)); log.flush()
    model.learn(total_timesteps=a.timesteps, callback=[Curriculum(), Budget()])
    sb3_to_mlp(model, os.path.join(a.outdir,
                                   f"ckpt_seed{a.seed}_step{a.timesteps:09d}.npz"), hidden)
    venv.close()


if __name__ == "__main__":
    main()
