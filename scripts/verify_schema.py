"""Verify the new HDF5 data schema from trajectories_v2.h5."""
import argparse
import numpy as np
import h5py

EXPECTED_DATASETS = ["latents", "actions", "rewards", "terminated", "truncated",
                     "success", "ee_pos", "cube_pos", "phase"]
EXPECTED_ATTRS    = ["episode_id", "length", "completed", "task", "seed", "timestamp"]

def verify(hdf5_path: str):
    with h5py.File(hdf5_path, "r") as f:
        episode_keys = sorted(k for k in f.keys() if k.startswith("episode_"))
        print(f"File: {hdf5_path}")
        print(f"Total episodes: {len(episode_keys)}\n")

        # ── Per-episode spot check (first, middle, last) ──────────────────────
        for key in [episode_keys[0], episode_keys[len(episode_keys) // 2], episode_keys[-1]]:
            grp = f[key]
            T   = int(grp.attrs["length"])
            print(f"--- {key} ---")

            # Check all expected datasets present
            missing = [d for d in EXPECTED_DATASETS if d not in grp]
            if missing:
                print(f"  MISSING datasets: {missing}")
            else:
                print(f"  datasets: OK ({len(EXPECTED_DATASETS)} present)")

            # Check all expected attrs present
            missing_attrs = [a for a in EXPECTED_ATTRS if a not in grp.attrs]
            if missing_attrs:
                print(f"  MISSING attrs: {missing_attrs}")
            else:
                print(f"  attrs:    OK  episode_id={grp.attrs['episode_id']}  "
                      f"length={T}  completed={grp.attrs['completed']}")

            # Shape / dtype checks
            assert grp["latents"].shape    == (T, 16, 8, 8),  f"latents shape wrong: {grp['latents'].shape}"
            assert grp["actions"].shape    == (T, 4),          f"actions shape wrong: {grp['actions'].shape}"
            assert grp["rewards"].shape    == (T,),            f"rewards shape wrong"
            assert grp["terminated"].shape == (T,),            f"terminated shape wrong"
            assert grp["truncated"].shape  == (T,),            f"truncated shape wrong"
            assert grp["success"].shape    == (T,),            f"success shape wrong"
            assert grp["ee_pos"].shape     == (T, 3),          f"ee_pos shape wrong"
            assert grp["cube_pos"].shape   == (T, 3),          f"cube_pos shape wrong"
            assert grp["phase"].shape      == (T,),            f"phase shape wrong"
            print(f"  shapes:   OK")

            # Dtype checks
            assert grp["latents"].dtype    == np.float16, f"latents dtype wrong: {grp['latents'].dtype}"
            assert grp["actions"].dtype    == np.float32, f"actions dtype wrong"
            assert grp["rewards"].dtype    == np.float32, f"rewards dtype wrong"
            assert grp["terminated"].dtype == np.bool_,   f"terminated dtype wrong: {grp['terminated'].dtype}"
            assert grp["truncated"].dtype  == np.bool_,   f"truncated dtype wrong"
            assert grp["success"].dtype    == np.bool_,   f"success dtype wrong"
            assert grp["ee_pos"].dtype     == np.float32, f"ee_pos dtype wrong"
            assert grp["cube_pos"].dtype   == np.float32, f"cube_pos dtype wrong"
            print(f"  dtypes:   OK")

            # Sanity: at least one of terminated/truncated must be True per episode
            term = grp["terminated"][:]
            trunc = grp["truncated"][:]
            assert (term | trunc).any(), "Neither terminated nor truncated ever True"
            print(f"  rewards:  min={grp['rewards'][:].min():.3f}  max={grp['rewards'][:].max():.3f}")
            print(f"  terminated: {term.sum()} steps  truncated: {trunc.sum()} steps  "
                  f"success: {grp['success'][:].sum()} steps")
            print(f"  phases:   {np.unique(grp['phase'][:]).tolist()}")
            print()

        # ── Dataset-wide summary ──────────────────────────────────────────────
        total_steps  = sum(f[k].attrs["length"]        for k in episode_keys)
        n_terminated = sum(f[k]["terminated"][:].any() for k in episode_keys)
        n_truncated  = sum(f[k]["truncated"][:].any()  for k in episode_keys)
        n_success    = sum(f[k]["success"][:].any()    for k in episode_keys)
        n            = len(episode_keys)

        print("=" * 45)
        print(f"  Episodes:   {n}")
        print(f"  Steps:      {total_steps:,}")
        print(f"  Terminated: {n_terminated:>4d}  ({n_terminated/n:.1%})")
        print(f"  Truncated:  {n_truncated:>4d}  ({n_truncated/n:.1%})")
        print(f"  Success:    {n_success:>4d}  ({n_success/n:.1%})")
        print("=" * 45)
        print("Schema verification PASSED")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5", default="trajectories_v2.h5")
    verify(p.parse_args().hdf5)
