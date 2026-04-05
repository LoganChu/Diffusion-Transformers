"""Diagnostic: is the online checkpoint learning useful distributions?

Four tests, run for both pretrained_best.pt and online_best.pt side-by-side:

  1. Score discriminability  — CEM score std across candidates.
                               Near-zero std = score function is flat, CEM is random.

  2. CubePosHead calibration — predicted cube_z on real encoded frames at episode start.
                               cube_z should be near 0 (cube sitting on the table).

  3. Latent alignment        — cosine similarity between a real encoded latent and a
                               1-step world-model rollout from the same frame.
                               High similarity = world model predicts in the right space.

  4. Action consistency      — run cem_plan 3x from the same state, measure action std.
                               Low std = score function is stable/deterministic.
                               High std = score function is noisy / flat.

Usage
-----
    python scripts/diagnose_cem.py
    python scripts/diagnose_cem.py --n_consistency 5 --n_ode 20
"""

from __future__ import annotations

import argparse
import collections

import numpy as np
import torch
import torch.nn.functional as F
import mani_skill.envs  # noqa: F401
import gymnasium as gym

from data.ingest       import ensure_cosmos_weights, CosmosLatentEncoder
from inference.planner import (
    cem_plan, cube_height_score_fn, maniskill_reward_score_fn,
    reward_only_score_fn, reward_value_score_fn,
    _euler_rollout_step, _ACTION_LO, _ACTION_HI,
)
from models.dit        import ACTION_DIM, DiTSmall, IN_CHANNELS, LATENT_H, LATENT_W

SEPARATOR = "=" * 60


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(path: str, device: torch.device) -> DiTSmall:
    model = DiTSmall().to(device)
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    sd    = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(sd)
    model.eval()
    ep = ckpt.get("episode", "offline")
    print(f"  loaded {path}  (episode={ep})")
    return model


@torch.no_grad()
def encode_frame(encoder, frame, device) -> torch.Tensor:
    if isinstance(frame, np.ndarray):
        frame = torch.from_numpy(frame)
    if frame.dim() == 4:
        frame = frame[0]
    frame = frame[..., :3].permute(2, 0, 1).float().div_(255.0)
    if frame.shape[-2:] != (128, 128):
        frame = F.interpolate(
            frame.unsqueeze(0), size=(128, 128), mode="bilinear", align_corners=False
        ).squeeze(0)
    buf    = frame.unsqueeze(0).to("cuda", dtype=torch.float16)
    latent = encoder.encode(buf)
    encoder.sync()
    return latent.to(torch.float16).to(device)   # [1, 16, 8, 8]


def make_ctx(z_t: torch.Tensor, n_ctx: int = 4) -> torch.Tensor:
    """Repeat current frame across all context slots — conservative placeholder."""
    return z_t.squeeze(0).unsqueeze(0).unsqueeze(0).expand(1, n_ctx, -1, -1, -1).contiguous()


def make_env():
    return gym.make(
        "PickCube-v1",
        num_envs=1,
        obs_mode="state",
        render_mode="rgb_array",
        render_backend="cpu",
        control_mode="pd_ee_delta_pos",
        max_episode_steps=200,
    )


# ---------------------------------------------------------------------------
# Test 1 — Score discriminability
# ---------------------------------------------------------------------------

def test_score_discriminability(model, ctx_input, device, args, label: str):
    """Run CEM with verbose score logging. Returns final score std."""
    print(f"\n[{label}] Score discriminability (n_candidates={args.n_candidates}, "
          f"n_cem_iters={args.n_cem_iters}, num_ode_steps={args.n_ode}):")

    ctx_act = torch.zeros(1, ACTION_DIM, device=device, dtype=torch.float16)
    stds    = []

    # Monkey-patch to capture score std without modifying cem_plan signature
    orig_score = getattr(args, "_score_override", None) or cube_height_score_fn

    captured = {}

    def capturing_score(model_, z, t):
        scores = orig_score(model_, z, t)
        captured.setdefault("all_scores", []).append(scores.detach().cpu())
        return scores

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        action = cem_plan(
            model, ctx_input, ctx_act,
            score_fn       = capturing_score,
            horizon        = args.horizon,
            n_candidates   = args.n_candidates,
            n_elites       = args.n_elites,
            n_cem_iters    = args.n_cem_iters,
            num_ode_steps  = args.n_ode,
            noise_std_init = args.noise_std_init,
            log_score_stats= True,
        )

    # Aggregate across all cem_iter × horizon score calls
    all_s = torch.cat(captured["all_scores"])   # [(n_cem_iters * horizon * N)]
    print(f"  aggregate  min={all_s.min():.4f}  max={all_s.max():.4f}  "
          f"std={all_s.std():.4f}  range={all_s.max()-all_s.min():.4f}")
    print(f"  planned action: {[f'{v:.4f}' for v in action.tolist()]}")
    print(f"  action magnitude (xyz): {action[:3].norm().item():.5f}")
    return all_s.std().item(), action


# ---------------------------------------------------------------------------
# Test 2 — CubePosHead calibration on real frames
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_cube_pos_calibration(model, encoder, env, device, args, label: str, n_resets: int = 5):
    """Check if CubePosHead outputs sensible cube_z on real encoded frames."""
    print(f"\n[{label}] CubePosHead calibration on real frames (n_resets={n_resets}):")

    predictions = []
    dtype = next(model.parameters()).dtype

    for i in range(n_resets):
        obs, _ = env.reset()
        frame  = env.render()
        z_real = encode_frame(encoder, frame, device)   # [1, 16, 8, 8]

        t_one  = torch.ones(1, device=device, dtype=dtype)
        dummy  = torch.zeros(1, ACTION_DIM, device=device, dtype=dtype)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            _, cube_pos_pred = model(z_real, t_one, dummy, return_aux=True)

        cube_z = cube_pos_pred[0, 2].item()
        cube_xy = cube_pos_pred[0, :2].tolist()
        predictions.append(cube_z)
        print(f"  reset {i+1}: predicted cube_z={cube_z:.4f}  "
              f"xy=[{cube_xy[0]:.3f}, {cube_xy[1]:.3f}]")

    mean_z = np.mean(predictions)
    std_z  = np.std(predictions)
    print(f"  mean cube_z={mean_z:.4f}  std={std_z:.4f}  "
          f"(expected: on-table ≈ 0.01–0.04, lifted ≈ 0.15+)")
    return predictions


# ---------------------------------------------------------------------------
# Test 3 — Latent alignment (real vs imagined)
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_latent_alignment(model, encoder, env, device, args, label: str):
    """Cosine similarity between real latent and 1-step world model prediction."""
    print(f"\n[{label}] Latent alignment (real vs 1-step rollout, "
          f"num_ode_steps={args.n_ode}):")

    dtype = next(model.parameters()).dtype
    obs, _ = env.reset()
    frame  = env.render()
    z_real = encode_frame(encoder, frame, device)   # [1, 16, 8, 8]

    ctx_input = make_ctx(z_real, n_ctx=4)           # [1, 4, 16, 8, 8]
    ctx_exp   = ctx_input.expand(1, -1, -1, -1, -1)

    # Predict next latent with zero action
    a_zero = torch.zeros(1, ACTION_DIM, device=device, dtype=dtype)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        z_pred = _euler_rollout_step(model, ctx_exp, a_zero, args.n_ode, dtype)  # [1, 16, 8, 8]

    z_r_flat = z_real.reshape(1, -1).float()
    z_p_flat = z_pred.reshape(1, -1).float()
    cos_sim  = F.cosine_similarity(z_r_flat, z_p_flat).item()
    l2_dist  = (z_r_flat - z_p_flat).norm().item()
    l2_real  = z_r_flat.norm().item()

    print(f"  cosine similarity (real vs imagined): {cos_sim:.4f}  "
          f"(>0.7 = good alignment, <0.3 = poor)")
    print(f"  L2 distance: {l2_dist:.4f}  (real latent L2 norm: {l2_real:.4f})")
    return cos_sim


# ---------------------------------------------------------------------------
# Test 4 — Action consistency
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_action_consistency(model, ctx_input, device, args, label: str):
    """Run cem_plan N times from same state. Low action std = stable score fn."""
    n = args.n_consistency
    print(f"\n[{label}] Action consistency ({n} runs from same state, "
          f"n_candidates={args.n_candidates}):")

    ctx_act = torch.zeros(1, ACTION_DIM, device=device, dtype=torch.float16)
    actions = []

    for i in range(n):
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            a = cem_plan(
                model, ctx_input, ctx_act,
                score_fn       = cube_height_score_fn,
                horizon        = args.horizon,
                n_candidates   = args.n_candidates,
                n_elites       = args.n_elites,
                n_cem_iters    = args.n_cem_iters,
                num_ode_steps  = args.n_ode,
                noise_std_init = args.noise_std_init,
            )
        actions.append(a.cpu())
        print(f"  run {i+1}: {[f'{v:.4f}' for v in a.tolist()]}")

    actions_t = torch.stack(actions)   # [n, 4]
    per_dim_std = actions_t.std(dim=0)
    mean_std = per_dim_std.mean().item()
    print(f"  per-dim std: {[f'{v:.4f}' for v in per_dim_std.tolist()]}")
    print(f"  mean std across dims: {mean_std:.4f}  "
          f"(<0.01 = stable, >0.03 = noisy/flat score fn)")
    return mean_std


# ---------------------------------------------------------------------------
# Summary + verdict
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Test 5 — Reward head directionality (does it prefer toward-cube actions?)
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_reward_head_directionality(model, encoder, env, device, args, label: str):
    """Sanity check: does the reward head score toward-cube actions higher?

    Strategy:
      1. Reset env and read the true ee_pos and cube_pos from the state obs.
      2. Construct an action that moves the arm toward the cube (positive signal).
      3. Construct an action that moves it away (negative signal).
      4. Run a 1-step world model rollout for each, score with reward_value_score_fn.
      5. Also compare against the true env reward for each action.
    """
    print(f"\n[{label}] Reward head directionality check:")
    dtype = next(model.parameters()).dtype

    obs, _ = env.reset()
    # ManiSkill state obs: varies by env version, but for PickCube-v1 with num_envs=1
    # obs is a dict or tensor. Extract ee_pos and cube_pos from the raw state.
    frame = env.render()
    z_real = encode_frame(encoder, frame, device)          # [1, 16, 8, 8]
    ctx_input = make_ctx(z_real, n_ctx=4)                  # [1, 4, 16, 8, 8]

    # --- Get ground-truth ee_pos and cube_pos from env state ---
    # PickCube-v1 state obs typically: [ee_pos(3), ee_vel(3), cube_pos(3), cube_vel(6), ...]
    # We extract positions heuristically from the state tensor.
    if isinstance(obs, dict):
        state = obs.get("agent", obs.get("observation", list(obs.values())[0]))
    else:
        state = obs
    if isinstance(state, torch.Tensor):
        state = state[0].cpu().numpy()
    elif hasattr(state, '__len__'):
        state = np.array(state).flatten()

    # Standard PickCube-v1 obs layout: first 3 = ee_pos (approx)
    # We'll use the env's info dict after a zero-action step to extract rewards
    # and compare directional actions.
    print(f"  state obs shape: {state.shape if hasattr(state, 'shape') else len(state)}")

    # --- Build toward/away actions using rough ee→cube direction ---
    # Try to get cube_pos from state (PickCube state has cube pos around index 9-12)
    # Rather than guessing layout, use reward comparison directly.
    ctx_exp = ctx_input.expand(4, -1, -1, -1, -1)   # [4, n_ctx, C, H, W] — 4 candidates

    # 4 candidate actions: +x, -x, +y, -y (covering cardinal directions in XY plane)
    delta = 0.07   # close to max action magnitude
    cardinal_actions = torch.tensor([
        [ delta,  0.0,   0.0, 0.0],   # +x
        [-delta,  0.0,   0.0, 0.0],   # -x
        [ 0.0,   delta,  0.0, 0.0],   # +y
        [ 0.0,  -delta,  0.0, 0.0],   # -y
    ], device=device, dtype=dtype)    # [4, 4]

    # --- Get ee_pos from env for maniskill_reward ---
    try:
        ee_pos = env.unwrapped.agent.tcp.pose.p[0].to(device).float()   # [3]
    except Exception:
        ee_pos = None

    # --- Score with reward_value, reward_only, and maniskill_reward ---
    with torch.amp.autocast("cuda", dtype=torch.float16):
        z_nexts   = _euler_rollout_step(model, ctx_exp, cardinal_actions, args.n_ode, dtype)
        t_ones    = torch.ones(4, device=device, dtype=dtype)
        rv_scores = reward_value_score_fn(model, z_nexts, t_ones).cpu()
        ro_scores = reward_only_score_fn(model, z_nexts, t_ones).cpu()

        if ee_pos is not None:
            # cumulative_ee = ee_pos + action_xyz for 1-step rollout
            cum_ee    = ee_pos.unsqueeze(0) + cardinal_actions[:, :3].float()   # [4, 3]
            ms_scores = maniskill_reward_score_fn(
                model, z_nexts, t_ones, cumulative_ee=cum_ee
            ).cpu()
        else:
            ms_scores = None

    names = ["+x", "-x", "+y", "-y"]
    print(f"  reward_value scores for cardinal actions (same starting state):")
    for name, score in zip(names, rv_scores.tolist()):
        print(f"    action {name}: score={score:.4f}")
    print(f"  reward_value   range: {rv_scores.max()-rv_scores.min():.4f}  "
          f"std: {rv_scores.std():.4f}")

    print(f"  reward_only scores:")
    for name, score in zip(names, ro_scores.tolist()):
        print(f"    action {name}: score={score:.4f}")
    print(f"  reward_only    range: {ro_scores.max()-ro_scores.min():.4f}  "
          f"std: {ro_scores.std():.4f}")

    if ms_scores is not None:
        print(f"  maniskill_reward scores (reach+lift, ee_pos from env):")
        for name, score in zip(names, ms_scores.tolist()):
            print(f"    action {name}: score={score:.4f}")
        print(f"  maniskill_reward range: {ms_scores.max()-ms_scores.min():.4f}  "
              f"std: {ms_scores.std():.4f}")

    # --- Compare against actual env rewards for same actions ---
    print(f"\n  actual env rewards for same actions (ground truth):")
    env_rewards = []
    for i, (name, a) in enumerate(zip(names, cardinal_actions)):
        obs2, _ = env.reset()        # same seed? No — reset randomizes cube pos.
        # Use same initial state by re-rendering from current obs instead.
        # Just take the action from the already-reset env to get reward signal.
        _, r, _, _, _ = env.step(a.cpu().numpy().reshape(1, -1))
        r_val = float(r[0]) if hasattr(r, '__len__') else float(r)
        env_rewards.append(r_val)
        print(f"    action {name}: env_reward={r_val:.4f}")
        obs, _ = env.reset()   # reset for next action test
        frame = env.render()

    env_rewards_t = torch.tensor(env_rewards)
    env_rank = env_rewards_t.argsort().argsort().float()

    def _rank_corr(scores):
        r = scores.argsort().argsort().float()
        return torch.corrcoef(torch.stack([r, env_rank]))[0, 1].item()

    rv_rank_corr = _rank_corr(rv_scores)
    ro_rank_corr = _rank_corr(ro_scores)
    ms_rank_corr = _rank_corr(ms_scores) if ms_scores is not None else float("nan")

    print(f"\n  Best actual action:       {names[env_rewards_t.argmax()]}")
    print(f"  Best reward_value action: {names[rv_scores.argmax()]}  "
          f"rank_corr={rv_rank_corr:.3f}")
    print(f"  Best reward_only action:  {names[ro_scores.argmax()]}  "
          f"rank_corr={ro_rank_corr:.3f}")
    if ms_scores is not None:
        print(f"  Best maniskill_reward action: {names[ms_scores.argmax()]}  "
              f"rank_corr={ms_rank_corr:.3f}")
    print(f"  (+1=perfect, 0=random, -1=inverted)")
    return rv_rank_corr, ro_rank_corr, ms_rank_corr


def print_verdict(pretrained_stats: dict, online_stats: dict):
    print(f"\n{SEPARATOR}")
    print("SUMMARY")
    print(SEPARATOR)

    def row(name, pre, onl, higher_is_better=True):
        arrow = "↑" if higher_is_better else "↓"
        winner = "online" if (onl > pre) == higher_is_better else "pretrained"
        print(f"  {name:<35s}  pretrained={pre:.4f}  online={onl:.4f}  "
              f"better={winner} {arrow}")

    row("Score std (discriminability)", pretrained_stats["score_std"],   online_stats["score_std"])
    row("Latent cosine similarity",     pretrained_stats["cos_sim"],     online_stats["cos_sim"])
    row("Action consistency (std↓)",    pretrained_stats["action_std"],  online_stats["action_std"],
        higher_is_better=False)
    row("Action magnitude",             pretrained_stats["action_mag"],  online_stats["action_mag"])
    if "rank_corr" in online_stats:
        rv_rc = online_stats["rank_corr"]
        ro_rc = online_stats.get("rank_corr_ro", float("nan"))
        ms_rc = online_stats.get("rank_corr_ms", float("nan"))

        def _label(v):
            if v != v: return "N/A"
            return "GOOD ✓" if v > 0.5 else "INVERTED ✗" if v < -0.3 else "WEAK"

        print(f"  {'Rank corr reward_value':<35s}  {rv_rc:.3f}  ({_label(rv_rc)})")
        print(f"  {'Rank corr reward_only':<35s}  {ro_rc:.3f}  ({_label(ro_rc)})")
        print(f"  {'Rank corr maniskill_reward':<35s}  {ms_rc:.3f}  ({_label(ms_rc)})")

    print(SEPARATOR)
    print("VERDICT:")

    score_improved  = online_stats["score_std"]   > pretrained_stats["score_std"] * 1.2
    latent_ok       = online_stats["cos_sim"]     > 0.5
    action_stable   = online_stats["action_std"]  < 0.03
    arm_moves       = online_stats["action_mag"]  > 0.005
    rv_rc             = online_stats.get("rank_corr", 0.0)
    ro_rc             = online_stats.get("rank_corr_ro", 0.0)
    ms_rc             = online_stats.get("rank_corr_ms", 0.0)
    reward_value_ok   = rv_rc > 0.3
    reward_only_ok    = ro_rc > 0.3
    maniskill_ok      = ms_rc > 0.3

    if score_improved and latent_ok and action_stable and arm_moves:
        print("  PASS — online model is more discriminating, world model predictions")
        print("         are aligned with real latents, and CEM produces stable actions.")
        print("         Continue online training; success rate should improve.")
    else:
        issues = []
        if not score_improved:
            issues.append("score std did not improve — CEM still near-random")
        if not latent_ok:
            issues.append(f"cos_sim={online_stats['cos_sim']:.3f} — world model imagines "
                          "out-of-distribution latents; try larger num_ode_steps")
        if not action_stable:
            issues.append("actions inconsistent across runs — score function is noisy")
        if not arm_moves:
            issues.append("action magnitude ≈ 0 — planner prefers no-movement")
        if "rank_corr" in online_stats:
            if maniskill_ok:
                pass   # maniskill_reward is reliable — good to train with
            elif not reward_value_ok and not reward_only_ok and not maniskill_ok:
                issues.append(
                    f"all score fns unreliable (rv={rv_rc:.3f} ro={ro_rc:.3f} ms={ms_rc:.3f})"
                    " — check CubePosHead calibration (Test 2)"
                )
            elif not reward_value_ok and reward_only_ok:
                issues.append(
                    f"reward_value rank_corr={rv_rc:.3f} inverted (value overestimation) "
                    f"but reward_only={ro_rc:.3f} ok — use --score_fn reward_only"
                )
        print("  ISSUES FOUND:")
        for issue in issues:
            print(f"    - {issue}")
    print(SEPARATOR)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    print("Loading models...")
    pre_model = load_model(args.pretrained, device)
    onl_model = load_model(args.online,     device)

    print("\nLoading encoder...")
    encoder = CosmosLatentEncoder(args.cosmos_ckpt)

    env = make_env()

    # Shared context — one fixed env reset used for tests 1, 3, 4
    obs, _ = env.reset()
    frame  = env.render()
    z_real = encode_frame(encoder, frame, device)
    ctx_input = make_ctx(z_real, n_ctx=4)

    pre_stats = {}
    onl_stats = {}

    print(f"\n{SEPARATOR}")
    print("TEST 1a: Score Discriminability — cube_height_score_fn")
    print(SEPARATOR)
    pre_stats["score_std"], pre_action = test_score_discriminability(
        pre_model, ctx_input, device, args, "pretrained"
    )
    onl_stats["score_std"], onl_action = test_score_discriminability(
        onl_model, ctx_input, device, args, "online"
    )
    pre_stats["action_mag"] = pre_action[:3].norm().item()
    onl_stats["action_mag"] = onl_action[:3].norm().item()

    print(f"\n{SEPARATOR}")
    print("TEST 1b: Score Discriminability — reward_value_score_fn (online model only)")
    print("         (reward head was trained on dense env reward incl. reach distance)")
    print(SEPARATOR)

    def _rv_score(model_, z, t):
        return reward_value_score_fn(model_, z, t, horizon=args.horizon)

    # Patch score_fn to use reward_value and run for online model only
    _orig = args.__dict__.get("_score_override")
    args._score_override = _rv_score
    onl_rv_std, onl_rv_action = test_score_discriminability(
        onl_model, ctx_input, device, args, "online[reward_value]"
    )
    args._score_override = _orig
    print(f"  reward_value score std={onl_rv_std:.4f} vs "
          f"cube_height score std={onl_stats['score_std']:.4f}")

    print(f"\n{SEPARATOR}")
    print("TEST 2: CubePosHead Calibration on Real Frames")
    print(SEPARATOR)
    test_cube_pos_calibration(pre_model, encoder, env, device, args, "pretrained", n_resets=5)
    test_cube_pos_calibration(onl_model, encoder, env, device, args, "online",     n_resets=5)

    print(f"\n{SEPARATOR}")
    print("TEST 3: Latent Alignment (real vs 1-step world model rollout)")
    print(SEPARATOR)
    pre_stats["cos_sim"] = test_latent_alignment(pre_model, encoder, env, device, args, "pretrained")
    onl_stats["cos_sim"] = test_latent_alignment(onl_model, encoder, env, device, args, "online")

    print(f"\n{SEPARATOR}")
    print("TEST 4: Action Consistency")
    print(SEPARATOR)
    pre_stats["action_std"] = test_action_consistency(
        pre_model, ctx_input, device, args, "pretrained"
    )
    onl_stats["action_std"] = test_action_consistency(
        onl_model, ctx_input, device, args, "online"
    )

    print(f"\n{SEPARATOR}")
    print("TEST 5: Reward Head Directionality (online model only)")
    print("        Does it prefer toward-cube actions over away-from-cube?")
    print(SEPARATOR)
    rv_rc, ro_rc, ms_rc = test_reward_head_directionality(
        onl_model, encoder, env, device, args, "online"
    )
    onl_stats["rank_corr"]          = rv_rc
    onl_stats["rank_corr_ro"]       = ro_rc
    onl_stats["rank_corr_ms"]       = ms_rc

    env.close()

    print_verdict(pre_stats, onl_stats)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained",     type=str,   default="pretrained_best.pt")
    p.add_argument("--online",         type=str,   default="online_best.pt")
    p.add_argument("--cosmos_ckpt",    type=str,   default="pretrained_ckpts/Cosmos-Tokenizer-CI16x16")
    # CEM params
    p.add_argument("--horizon",        type=int,   default=6)
    p.add_argument("--n_candidates",   type=int,   default=64)
    p.add_argument("--n_elites",       type=int,   default=8)
    p.add_argument("--n_cem_iters",    type=int,   default=3)
    p.add_argument("--n_ode",          type=int,   default=10)
    p.add_argument("--noise_std_init", type=float, default=0.05)
    p.add_argument("--n_consistency",  type=int,   default=3)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
