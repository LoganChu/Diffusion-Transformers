import h5py
import numpy as np

with h5py.File("trajectories_7k.h5", "r") as f:
    bad_ep = None
    clean_ep = None
    for key in sorted(f.keys()):
        if key == "metadata":
            continue
        grp = f[key]
        latents = grp["latents"][:].astype(np.float32)
        is_clean = np.isfinite(latents).all()
        if bad_ep is None and not is_clean:
            n_bad = (~np.isfinite(latents).all(axis=(1,2,3))).sum()
            bad_ep = (key, grp.attrs.get("seed", "?"), latents.shape[0], int(n_bad))
        if clean_ep is None and is_clean:
            clean_ep = (key, grp.attrs.get("seed", "?"), latents.shape[0])
        if bad_ep and clean_ep:
            break

    print(f"BAD episode:   key={bad_ep[0]}  seed={bad_ep[1]}  frames={bad_ep[2]}  bad_frames={bad_ep[3]}")
    print(f"CLEAN episode: key={clean_ep[0]}  seed={clean_ep[1]}  frames={clean_ep[2]}")
