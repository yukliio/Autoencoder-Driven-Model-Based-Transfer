import json
import numpy as np
from collections import defaultdict
import os

print(f"Current directory: {os.getcwd()}")

data_path = '../data/raw/agent_data_log.jsonl'
print(f"Looking for file at: {os.path.abspath(data_path)}")

if not os.path.exists(data_path):
    print(f"ERROR: File not found at {data_path}")
    exit(1)

with open(data_path, 'r') as f:
    lines = f.readlines()

print(f"Read {len(lines)} lines from file")

# Parse JSON
records = []
for i, line in enumerate(lines):
    if line.strip():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Warning: Couldn't parse line {i+1}: {e}")

print(f"Successfully parsed {len(records)} records")

# Group by episode_id
episodes = defaultdict(list)
for rec in records:
    episodes[rec['episode_id']].append(rec)

print(f"Found {len(episodes)} unique episodes")

# Build trajectories with chunking for long episodes
CHUNK_SIZE = 500
trajectories = []

for ep_id, steps in episodes.items():
    steps = sorted(steps, key=lambda x: x['step'])
    
    # Build trajectory
    full_traj = []
    for step_data in steps:
        obs = step_data['obs']
        action = step_data['action']
        combined = obs + action
        full_traj.append(combined)
    
    # Split into chunks if too long
    if len(full_traj) > CHUNK_SIZE:
        for i in range(0, len(full_traj), CHUNK_SIZE):
            chunk = full_traj[i:i+CHUNK_SIZE]
            if len(chunk) >= 50:  # Only keep chunks with at least 50 steps
                trajectories.append(chunk)
    else:
        trajectories.append(full_traj)

print(f"Created {len(trajectories)} trajectory chunks")

if len(trajectories) == 0:
    print("ERROR: No trajectories found!")
    exit(1)

# Pad to same length
max_len = max(len(t) for t in trajectories)
print(f"Max sequence length: {max_len}")

padded_trajs = []
for traj in trajectories:
    arr = np.array(traj)
    pad_width = ((0, max_len - len(traj)), (0, 0))
    arr = np.pad(arr, pad_width, constant_values=0)
    padded_trajs.append(arr)
padded_trajs = np.stack(padded_trajs)

# Normalize
mean = padded_trajs.mean(axis=(0,1), keepdims=True)
std = padded_trajs.std(axis=(0,1), keepdims=True) + 1e-8
norm_trajs = (padded_trajs - mean) / std

# Save
np.save('../data/processed/intersectionzoo_latent_mbtldata.npy', norm_trajs)
np.save('../data/processed/norm_mean.npy', mean)
np.save('../data/processed/norm_std.npy', std)

print("\n✓ Preprocessed data written to the 'processed/' folder.")
print(f"✓ Trajectories: {len(trajectories)}, sequence length: {max_len}, feature dimension: {norm_trajs.shape[2]}")