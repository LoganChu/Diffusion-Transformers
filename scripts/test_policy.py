"""Quick success-rate check for the guided policy."""
import sys, torch
sys.path.insert(0, '.')
import gymnasium as gym
import mani_skill.envs  # noqa: F401
from data.ingest import GuidedPolicy

env = gym.make('PickCube-v1', num_envs=1, obs_mode='state', render_mode='rgb_array',
               render_backend='cpu', control_mode='pd_ee_delta_pos', max_episode_steps=400)
policy = GuidedPolicy(env=env, num_envs=1, noise_scale=0.03, device='cpu')

results = []
for seed in range(20):
    obs, _ = env.reset(seed=seed)
    policy.phases.zero_()
    policy.phase_steps.zero_()
    max_phase = 0
    for step in range(400):
        action = policy()
        max_phase = max(max_phase, int(policy.phases[0].item()))
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated.any() or truncated.any():
            raw = info.get('success', False)
            success = bool(raw[0].item() if hasattr(raw, '__len__') else raw)
            outcome = 'TERM' if terminated.any() else 'TRUNC'
            results.append((seed, step + 1, outcome, success, max_phase))
            break
    else:
        results.append((seed, 400, 'TIMEOUT', False, max_phase))

env.close()
phase_names = {0: 'APPROACH', 1: 'DESCEND', 2: 'GRASP', 3: 'LIFT', 4: 'HOLD'}
for r in results:
    print(f'seed={r[0]:2d}  steps={r[1]:3d}  {r[2]}  success={r[3]}  max_phase={phase_names[r[4]]}')
print(f'\nSuccess rate: {sum(r[3] for r in results)}/{len(results)}')
