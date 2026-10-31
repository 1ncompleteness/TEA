#!/usr/bin/env python3
"""Cache LIBERO demo data from hdf5 to npy arrays for TEA Q-net training.

Extracts:
  proprio[N, 15]: ee_pos(3) + ee_ori(3) + gripper_states(2) + joint_states(7)
  actions[N, 7]: 7-DOF EEF delta
  episode_ends[M]: frame indices marking episode boundaries (success=done terminal)
"""

import argparse, os, glob, numpy as np, h5py


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.hdf5_dir, "*.hdf5")))
    if not files:
        raise SystemExit(f"No hdf5 in {args.hdf5_dir}")

    all_proprio = []
    all_actions = []
    episode_ends = []  # frame index where episode ends (success)
    frame_counter = 0

    for fi, path in enumerate(files):
        with h5py.File(path, 'r') as f:
            for demo_key in sorted(f['data'].keys(), key=lambda k: int(k.split('_')[1])):
                demo = f['data'][demo_key]
                T = demo['actions'].shape[0]
                ee_pos = demo['obs']['ee_pos'][:]       # [T, 3]
                ee_ori = demo['obs']['ee_ori'][:]       # [T, 3]
                grip = demo['obs']['gripper_states'][:]  # [T, 2]
                joints = demo['obs']['joint_states'][:]  # [T, 7]
                proprio = np.concatenate([ee_pos, ee_ori, grip, joints], axis=1)  # [T, 15]
                acts = demo['actions'][:]  # [T, 7]
                all_proprio.append(proprio.astype(np.float32))
                all_actions.append(acts.astype(np.float32))
                frame_counter += T
                episode_ends.append(frame_counter - 1)
        print(f"[{fi+1}/{len(files)}] {os.path.basename(path)}: {frame_counter} total frames so far")

    obs = np.concatenate(all_proprio, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    ends = np.array(episode_ends, dtype=np.int64)
    print(f"\nTotal: {len(obs)} frames, {len(ends)} episodes")
    print(f"proprio dim: {obs.shape[1]}, action dim: {actions.shape[1]}")

    np.save(os.path.join(args.out_dir, 'obs.npy'), obs)
    np.save(os.path.join(args.out_dir, 'actions.npy'), actions)
    np.save(os.path.join(args.out_dir, 'episode_ends.npy'), ends)
    print(f"Saved to {args.out_dir}")


if __name__ == '__main__':
    main()
