"""Merge HDF5 trajectory shards into a single file.

Usage:
    python scripts/merge_shards.py --shard_dir trajectories_shards --output trajectories_20k.h5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def merge(shard_dir: str, output_path: str) -> None:
    shards = sorted(Path(shard_dir).glob("shard_*.h5"))
    if not shards:
        raise FileNotFoundError(f"No shard_*.h5 files found in {shard_dir}")

    print(f"Merging {len(shards)} shards → {output_path}")

    with h5py.File(output_path, "w", rdcc_nbytes=64 * 1024 * 1024) as out:
        ep_idx = 0

        for shard_path in shards:
            print(f"  {shard_path.name} ... ", end="", flush=True)
            with h5py.File(shard_path, "r") as src:
                # copy metadata from the first shard only
                if ep_idx == 0 and "metadata" in src:
                    src.copy("metadata", out)

                eps_in_shard = 0
                for key in src.keys():
                    if key == "metadata":
                        continue
                    src.copy(key, out, name=f"episode_{ep_idx:04d}")
                    ep_idx += 1
                    eps_in_shard += 1

            print(f"{eps_in_shard} episodes")

        out.attrs["total_episodes"] = ep_idx

    # quick sanity check
    with h5py.File(output_path, "r") as f:
        keys = [k for k in f.keys() if k != "metadata"]
        total_steps = sum(f[k].attrs.get("length", 0) for k in keys)
        print(f"\nDone: {ep_idx} episodes, {total_steps:,} total steps → {output_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--shard_dir", default="trajectories_shards")
    p.add_argument("--output",    default="trajectories_20k.h5")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge(args.shard_dir, args.output)
