"""Debug policy phases — prints per-step state to understand early transitions."""
import sys
from pathlib import Path
import torch
import gymnasium as gym
import mani_skill.envs  # noqa: F401

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.ingest import GuidedPolicy

PHASE_NAMES = ["APPROACH", "DESCEND", "GRASP", "LIFT"]

env = gym.make(
    "PickCube-v1",
    num_envs=1,
    obs_mode="state",
    render_mode="rgb_array",
    render_backend="cpu",
    control_mode="pd_ee_delta_pos",
    max_episode_steps=210,
)

policy = GuidedPolicy(env=env, num_envs=1, noise_scale=0.0)  # no noise for debug

obs, _ = env.reset(seed=0)

for step in range(80):
    uwenv = env.unwrapped
    ee_pos   = uwenv.agent.tcp.pose.p[0]
    cube_pos = getattr(uwenv, policy._cube_attr).pose.p[0]

    hover_target = cube_pos.clone(); hover_target[2] = cube_pos[2] + policy.HOVER_HEIGHT
    near_target  = cube_pos.clone(); near_target[2]  = cube_pos[2] + policy.NEAR_HEIGHT

    dist_hover = (ee_pos - hover_target).norm().item()
    dist_near  = (ee_pos - near_target).norm().item()

    phase = policy.phases[0].item()
    pstep = policy.phase_steps[0].item()

    action = policy()
    obs, reward, terminated, truncated, _ = env.step(action)

    print(f"step {step:3d} | phase={PHASE_NAMES[phase]:<8} pstep={pstep:3d} | "
          f"ee=[{ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f}] "
          f"cube=[{cube_pos[0]:.3f},{cube_pos[1]:.3f},{cube_pos[2]:.3f}] | "
          f"d_hover={dist_hover:.3f} d_near={dist_near:.3f} | "
          f"reward={float(reward):.3f}")

    if terminated.any() or truncated.any():
        print("  >> episode ended")
        break

env.close()
