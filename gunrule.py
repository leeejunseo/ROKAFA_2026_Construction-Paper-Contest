"""기총 판정 변형 시험: 현재 위치 기준(현행) vs 리드점 기준(탄 비행시간 반영)."""
import sys, math, numpy as np, collections
sys.path.insert(0, sys.argv[1])
from multiprocessing import Pool
from dfxai.config import EngagementConfig
import dfxai.env as E
from dfxai.env import run_episode
from dfxai.agents.bt import BTPolicy
from dfxai.agents.bt_manual import ManualBTPolicy

V_MUZZLE = 1050.0          # M61 계열 탄속 근사 [m/s]
_orig = E.boresight_angle
_CUR = {}

def lead_angle(own, lx, ly, lz, r):
    env = _CUR['env']
    tgt = env.red if own is env.blue else env.blue
    tof = r / (V_MUZZLE + own.v)
    tv = tgt.vel
    fx, fy, fz = lx + tv[0] * tof, ly + tv[1] * tof, lz + tv[2] * tof
    return _orig(own, fx, fy, fz, math.sqrt(fx * fx + fy * fy + fz * fz))

_E = {}
def job(a):
    var, env6, s, side, opp = a
    key = (var, env6)
    if key not in _E:
        ec = EngagementConfig(**dict(var[1]))
        if env6:
            from dfxai.sixdof.env6dof import SixDofEnv
            _E[key] = SixDofEnv(ec=ec)
        else:
            _E[key] = E.DogfightEnv(ec=ec)
    env = _E[key]; _CUR['env'] = env
    E.boresight_angle = lead_angle if var[0] == 'lead' else _orig
    m = ManualBTPolicy(); o = BTPolicy(version=opp)
    r = run_episode(m, o, seed=s, env=env) if side == 0 else run_episode(o, m, seed=s, env=env)
    w = r.winner if side == 0 else -r.winner
    wez_m = r.wez_time_blue if side == 0 else r.wez_time_red
    wez_o = r.wez_time_red if side == 0 else r.wez_time_blue
    return var, env6, opp, (r.damage_dealt + r.damage_taken) > 0, r.outcome, w, wez_m, wez_o

if __name__ == '__main__':
    VARS = [('cur', ()), ('cur', (("cone_deg_start", 6.0), ("cone_deg_end", 12.0))),
            ('lead', ()), ('lead', (("cone_deg_start", 2.0), ("cone_deg_end", 6.0)))]
    n = int(sys.argv[2])
    jobs = [(v, e6, 10000 + i, sd, 2) for v in VARS for e6 in (False, True) for i in range(n) for sd in (0, 1)]
    with Pool(2) as p:
        res = p.map(job, jobs, chunksize=4)
    for v in VARS:
        for e6 in (False, True):
            x = [q for q in res if q[0] == v and q[1] == e6]
            oc = collections.Counter(q[4] for q in x)
            sc = np.mean([1 if q[5] == 1 else .5 if q[5] == 0 else 0 for q in x])
            print(f"{v[0]:4s} {str(dict(v[1])):45s} {'6DOF' if e6 else '3DOF'} 접촉 {np.mean([q[3] for q in x]):.2f} "
                  f"격추 {oc['gun_kill']/len(x):.2f} 충돌 {oc['collision']/len(x):.2f} BT-M점수 {sc:.3f} "
                  f"WEZ초 BT-M {np.mean([q[6] for q in x]):.2f} / BT-v2 {np.mean([q[7] for q in x]):.2f}")
