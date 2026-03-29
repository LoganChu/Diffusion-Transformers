import h5py
import numpy as np
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "trajectories_7k.h5"

with h5py.File(path, "r") as f:
    keys = [k for k in f.keys() if k != "metadata"]
    print(f"Total episodes: {len(keys)}")

    total_frames = 0
    bad_frames = 0
    clean_eps = 0

    for key in keys:
        grp = f[key]
        latents = grp["latents"][:].astype(np.float32)  # [T, 16, 8, 8]
        T = latents.shape[0]
        total_frames += T
        # bad frame = any NaN or inf in that frame
        bad_mask = ~np.isfinite(latents).all(axis=(1, 2, 3))  # [T]
        n_bad = bad_mask.sum()
        bad_frames += n_bad
        if n_bad == 0:
            clean_eps += 1

    print(f"Clean episodes (zero bad frames): {clean_eps}/{len(keys)}")
    print(f"Bad frames: {bad_frames:,} / {total_frames:,}  ({100*bad_frames/total_frames:.2f}%)")

    # Show latent value range for a clean episode
    for key in keys:
        grp = f[key]
        latents = grp["latents"][:].astype(np.float32)
        if np.isfinite(latents).all():
            print(f"\nClean episode {key} latent stats:")
            print(f"  min={latents.min():.4f}  max={latents.max():.4f}  "
                  f"mean={latents.mean():.4f}  std={latents.std():.4f}")
            break

    # Show raw float16 range for a bad episode
    for key in keys:
        grp = f[key]
        latents_f16 = grp["latents"][:]  # raw float16
        if not np.isfinite(latents_f16.astype(np.float32)).all():
            print(f"\nBad episode {key} raw float16 max abs: {np.abs(latents_f16[np.isfinite(latents_f16)]).max():.1f}")
            print(f"  n_inf: {np.isinf(latents_f16.astype(np.float32)).sum()}")
            print(f"  n_nan: {np.isnan(latents_f16.astype(np.float32)).sum()}")
            break
