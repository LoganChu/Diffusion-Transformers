"""Visualize sampled EE trajectories from trajectories_10k.h5 in 3D.

Actions stored are pd_ee_delta_pos: [dx, dy, dz, gripper].
Cumulatively summing (dx, dy, dz) reconstructs the relative EE path per episode.

Usage:
    conda run -n ml_env python scripts/viz_trajectories.py
    conda run -n ml_env python scripts/viz_trajectories.py --hdf5 trajectories_10k.h5 --n 50
    conda run -n ml_env python scripts/viz_trajectories.py --n 20 --color reward --out viz.html
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import h5py
import numpy as np

try:
    import plotly.graph_objects as go
except ImportError:
    print("plotly not found — install with: pip install plotly")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5",  default="trajectories_10k.h5",
                   help="Path to merged HDF5 file")
    p.add_argument("--n",     type=int, default=30,
                   help="Number of episodes to sample")
    p.add_argument("--color", choices=["reward", "phase", "gripper", "episode"],
                   default="phase",
                   help="What to color trajectories by")
    p.add_argument("--seed",  type=int, default=0)
    p.add_argument("--out",   default="outputs/viz_trajectories.html",
                   help="Output HTML file (opened in browser if --show)")
    p.add_argument("--show",  action="store_true", default=True,
                   help="Open browser after rendering (default True)")
    p.add_argument("--completed_only", action="store_true", default=False,
                   help="Only visualize episodes where the task was completed")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Phase heuristic from gripper signal
# ─────────────────────────────────────────────────────────────────────────────
def infer_phases(actions: np.ndarray) -> np.ndarray:
    """Rough 4-phase label from gripper column.

    GuidedPolicy uses:
      phase 0 (APPROACH) and 1 (DESCEND): gripper open  → gripper dim > 0
      phase 2 (GRASP) and 3 (LIFT):       gripper close → gripper dim < 0

    We add a split at the first close transition (grasp→lift) based on dz sign.
    Returns int array [T] in {0,1,2,3}.
    """
    gripper = actions[:, 3]          # last dim
    T = len(gripper)
    phases = np.zeros(T, dtype=np.int32)

    # find first step where gripper goes negative (closed)
    close_mask = gripper < 0
    if not close_mask.any():
        return phases  # all approach

    first_close = int(np.argmax(close_mask))

    # split open region at midpoint → approach / descend
    mid_open = first_close // 2
    phases[:mid_open] = 0       # APPROACH
    phases[mid_open:first_close] = 1  # DESCEND

    # split close region at first upward dz → grasp / lift
    close_region = actions[first_close:, 2]   # dz
    up_mask = close_region > 0.005
    if up_mask.any():
        first_up = int(np.argmax(up_mask))
        phases[first_close: first_close + first_up] = 2  # GRASP
        phases[first_close + first_up:] = 3              # LIFT
    else:
        phases[first_close:] = 2  # GRASP (never lifted)

    return phases


PHASE_COLORS = ["#4C9BE8", "#F5A623", "#E05C5C", "#6AC16A"]
PHASE_NAMES  = ["Approach", "Descend", "Grasp", "Lift"]


# ─────────────────────────────────────────────────────────────────────────────
# Load episodes
# ─────────────────────────────────────────────────────────────────────────────
def load_episodes(hdf5_path: str, n: int, seed: int, completed_only: bool):
    with h5py.File(hdf5_path, "r") as f:
        keys = [k for k in f.keys() if k.startswith("episode_")]

        if completed_only:
            keys = [k for k in keys if f[k].attrs.get("completed", True)]

        random.seed(seed)
        sampled = random.sample(keys, min(n, len(keys)))

        episodes = []
        for k in sampled:
            grp = f[k]
            actions = grp["actions"][:]          # [T, 4]  float32
            rewards = grp["rewards"][:]          # [T]     float32
            completed = bool(grp.attrs.get("completed", False))
            episodes.append({
                "key": k,
                "actions": actions,
                "rewards": rewards,
                "completed": completed,
            })

    print(f"Loaded {len(episodes)} episodes from {hdf5_path}")
    return episodes


# ─────────────────────────────────────────────────────────────────────────────
# Build traces
# ─────────────────────────────────────────────────────────────────────────────
def make_traces(episodes: list[dict], color_by: str) -> list:
    """Return a list of Plotly 3-D scatter traces."""
    traces = []
    n_ep = len(episodes)

    # colorscale for reward / episode coloring
    def episode_color(i):
        t = i / max(n_ep - 1, 1)
        r = int(255 * (1 - t))
        b = int(255 * t)
        return f"rgb({r},80,{b})"

    for ep_idx, ep in enumerate(episodes):
        actions   = ep["actions"]          # [T, 4]
        rewards   = ep["rewards"]          # [T]
        completed = ep["completed"]

        # ── reconstruct EE path ─────────────────────────────────────────────
        # actions[:, :3] = (dx, dy, dz) in world frame
        # We start at origin; real offset is unknown but relative shape is exact.
        xyz = np.cumsum(actions[:, :3], axis=0)   # [T, 3]

        phases = infer_phases(actions)             # [T]

        # ── choose color signal ──────────────────────────────────────────────
        if color_by == "phase":
            # One sub-trace per phase segment for clean legend
            for ph_id in range(4):
                mask = phases == ph_id
                if not mask.any():
                    continue
                traces.append(go.Scatter3d(
                    x=xyz[mask, 0],
                    y=xyz[mask, 1],
                    z=xyz[mask, 2],
                    mode="markers",
                    marker=dict(size=2, color=PHASE_COLORS[ph_id], opacity=0.6),
                    name=PHASE_NAMES[ph_id],
                    legendgroup=PHASE_NAMES[ph_id],
                    showlegend=(ep_idx == 0),  # show each phase label once
                ))
            # thin line for full trajectory
            traces.append(go.Scatter3d(
                x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                mode="lines",
                line=dict(color="rgba(150,150,150,0.2)", width=1),
                showlegend=False,
            ))

        elif color_by == "reward":
            cum_rew = np.cumsum(rewards)
            traces.append(go.Scatter3d(
                x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                mode="lines+markers",
                marker=dict(
                    size=2, color=cum_rew,
                    colorscale="Viridis", showscale=(ep_idx == 0),
                    colorbar=dict(title="Cumulative reward", thickness=15) if ep_idx == 0 else None,
                ),
                line=dict(color="rgba(150,150,150,0.3)", width=1),
                showlegend=False,
                name=ep["key"],
            ))

        elif color_by == "gripper":
            gripper = actions[:, 3]
            traces.append(go.Scatter3d(
                x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                mode="lines+markers",
                marker=dict(
                    size=2, color=gripper,
                    colorscale="RdYlGn", showscale=(ep_idx == 0),
                    colorbar=dict(title="Gripper", thickness=15) if ep_idx == 0 else None,
                ),
                line=dict(color="rgba(150,150,150,0.3)", width=1),
                showlegend=False,
                name=ep["key"],
            ))

        elif color_by == "episode":
            col = episode_color(ep_idx)
            label = f"{ep['key']} ({'✓' if completed else '✗'})"
            traces.append(go.Scatter3d(
                x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                mode="lines+markers",
                marker=dict(size=2, color=col, opacity=0.7),
                line=dict(color=col, width=2),
                name=label,
            ))

    return traces


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    hdf5_path = Path(args.hdf5)
    if not hdf5_path.exists():
        print(f"HDF5 not found: {hdf5_path}")
        sys.exit(1)

    episodes = load_episodes(
        str(hdf5_path), args.n, args.seed, args.completed_only
    )
    if not episodes:
        print("No episodes matched the filter.")
        sys.exit(1)

    traces = make_traces(episodes, args.color)

    n_completed = sum(e["completed"] for e in episodes)
    title = (
        f"EE Trajectories  |  {len(episodes)} episodes  "
        f"({n_completed} completed)  |  color={args.color}"
    )

    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            title=title,
            scene=dict(
                xaxis_title="X (m, cumulative Δ)",
                yaxis_title="Y (m, cumulative Δ)",
                zaxis_title="Z (m, cumulative Δ)",
                aspectmode="data",
            ),
            legend=dict(itemsizing="constant"),
            margin=dict(l=0, r=0, t=40, b=0),
            template="plotly_dark",
        ),
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out))
    print(f"Saved → {out.resolve()}")

    if args.show:
        import webbrowser
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
