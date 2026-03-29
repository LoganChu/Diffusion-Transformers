import h5py, sys
shard_dir = sys.argv[1] if len(sys.argv) > 1 else "trajectories_shards_10k"
target = int(sys.argv[2]) if len(sys.argv) > 2 else 625
total = 0
for i in range(16):
    path = f"{shard_dir}/shard_{i:02d}.h5"
    try:
        with h5py.File(path, "r") as f:
            n = len([k for k in f.keys() if k.startswith("episode_")])
        total += n
        print(f"shard_{i:02d}: {n:4d}/{target}")
    except Exception as e:
        print(f"shard_{i:02d}: locked/in-progress ({e})")
print(f"\nTotal: {total}/{16*target}  ({100*total/(16*target):.1f}%)")
