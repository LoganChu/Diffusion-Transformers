"""Test GuidedPolicy with pd_ee_delta_pos. Observe phase cycles and EE convergence."""
import warnings; warnings.filterwarnings("ignore")
import sys, torch, gymnasium as gym
import mani_skill.envs  # noqa
sys.path.insert(0, ".")
from data.ingest import GuidedPolicy

NUM_ENVS  = 4
MAX_STEPS = 300

env = gym.make("PickCube-v1", num_envs=NUM_ENVS, obs_mode="state",
               control_mode="pd_ee_delta_pos", max_episode_steps=200)
obs, _ = env.reset(seed=42)

policy = GuidedPolicy(env=env, num_envs=NUM_ENVS, noise_scale=0.15)
obs, _ = env.reset(seed=42)

total_rewards = torch.zeros(NUM_ENVS, device="cuda")

for step in range(MAX_STEPS):
    ee_pos, cube_pos = policy._get_ee_and_cube_pos()
    hover_tgt = cube_pos.clone(); hover_tgt[:, 2] += policy.HOVER_HEIGHT
    near_tgt  = cube_pos.clone(); near_tgt[:, 2]  += policy.NEAR_HEIGHT
    phase = policy.phases[0].item()
    active_tgt = hover_tgt if phase == 0 else near_tgt
    d_tgt  = torch.norm(ee_pos[0] - active_tgt[0]).item()
    d_cube = torch.norm(ee_pos[0] - cube_pos[0]).item()

    actions = policy()
    obs, rewards, terms, truncs, infos = env.step(actions)
    dones = (terms | truncs).bool()
    total_rewards += rewards

    if step % 5 == 0:
        phases = policy.phases.cpu().tolist()
        psteps = policy.phase_steps.cpu().tolist()
        rew    = rewards.cpu().tolist()
        print(f"s={step:3d} ph={phases} ps={psteps} "
              f"ee=[{ee_pos[0,0].item():.3f},{ee_pos[0,1].item():.3f},{ee_pos[0,2].item():.3f}] "
              f"d_tgt={d_tgt:.3f} d_cube={d_cube:.3f} "
              f"rew={[f'{r:.3f}' for r in rew]}")

    policy.reset_done_envs(dones)

print(f"\nTotal rewards: {[round(r,3) for r in total_rewards.cpu().tolist()]}")
print(f"Mean total reward: {total_rewards.mean().item():.4f}")
env.close()
