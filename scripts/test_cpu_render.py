"""Test render_mode='rgb_array' with CPU renderer backend (no Vulkan)."""
import gymnasium as gym
import mani_skill.envs  # noqa: F401

print("Creating env with render_mode='rgb_array' + render_backend='cpu'...")
env = gym.make(
    "PickCube-v1",
    num_envs=1,
    obs_mode="state",
    render_mode="rgb_array",
    render_backend="cpu",
)
print("env created OK")

obs, _ = env.reset(seed=0)
action = env.action_space.sample()
env.step(action)

frame = env.render()
if hasattr(frame, "cpu"):
    frame = frame.cpu().numpy()
print(f"render() OK — shape={frame.shape}  dtype={frame.dtype}")

env.close()
print("PASS: CPU renderer works")
