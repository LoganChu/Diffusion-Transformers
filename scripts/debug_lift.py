"""Check if cube gets lifted — print GRASP/LIFT phases and cube_z."""
import sys
from pathlib import Path
import torch
import gymnasium as gym
import mani_skill.envs  # noqa: F401
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.ingest import GuidedPolicy

PHASE_NAMES = ["APPROACH", "DESCEND", "GRASP", "LIFT"]

env = gym.make(
    "PickCube-v1", num_envs=1, obs_mode="state", render_mode="rgb_array",
    render_backend="cpu", control_mode="pd_ee_delta_pos", max_episode_steps=310,
)
policy = GuidedPolicy(env=env, num_envs=1, noise_scale=0.0)
obs, _ = env.reset(seed=0)

max_reward = 0.0
for step in range(300):
    uwenv = env.unwrapped
    ee_pos   = uwenv.agent.tcp.pose.p[0]
    cube_pos = getattr(uwenv, policy._cube_attr).pose.p[0]
    phase = policy.phases[0].item()
    action = policy()
    obs, reward, terminated, truncated, _ = env.step(action)
    r = float(reward)
    max_reward = max(max_reward, r)
    if phase >= 2:
        print(f"step {step:3d} | {PHASE_NAMES[phase]:<8} | "
              f"ee_z={float(ee_pos[2]):.3f} cube_z={float(cube_pos[2]):.3f} | reward={r:.3f}")
    if terminated.any() or truncated.any():
        print(f"  >> episode ended  success={terminated.any().item()}")
        break

print(f"max reward: {max_reward:.3f}")
env.close()
