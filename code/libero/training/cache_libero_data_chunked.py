#!/usr/bin/env python3
"""Cache LIBERO demos with chunked actions for Q-chunking (Li 2025, arxiv:2507.07969).

Outputs:
  proprio[N, 15]
  action_chunk[N, H, 7]            zero-padded at terminals
  action_chunk_mask[N, H]          1 where chunk-action is valid, 0 at pad
  next_proprio[N, 15]              s_{t+H}  (zero-padded with final state)
  reward_chunk[N, H]               per-step reward over chunk (sparse: +1 at success terminal)
  done[N]                          episode boundary flag
  episode_ends[M]

Usage:
  python cache_libero_data_chunked.py --hdf5_dir /workspace/libero_demos/libero_object \
                                      --out_dir /workspace/algorithm/data/libero_object_chunk5 \
                                      --chunk 5
"""
import argparse, os, glob, numpy as np, h5py


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--chunk", type=int, default=5)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    H = args.chunk

    files = sorted(glob.glob(os.path.join(args.hdf5_dir, "*.hdf5")))
    if not files:
        raise SystemExit(f"No hdf5 in {args.hdf5_dir}")

    proprios, action_chunks, masks, next_props, rew_chunks, dones = (
        [], [], [], [], [], [])
    episode_ends, frame_counter = [], 0

    for fi, path in enumerate(files):
        with h5py.File(path, 'r') as f:
            for demo_key in sorted(f['data'].keys(), key=lambda k: int(k.split('_')[1])):
                demo = f['data'][demo_key]
                T = demo['actions'].shape[0]
                ee_pos = demo['obs']['ee_pos'][:]
                ee_ori = demo['obs']['ee_ori'][:]
                grip = demo['obs']['gripper_states'][:]
                joints = demo['obs']['joint_states'][:]
                proprio = np.concatenate([ee_pos, ee_ori, grip, joints], axis=1).astype(np.float32)
                acts = demo['actions'][:].astype(np.float32)
                # Chunk each timestep t -> actions[t:t+H], zero-pad if overrun
                for t in range(T):
                    take = min(H, T - t)
                    ch = np.zeros((H, 7), dtype=np.float32)
                    mk = np.zeros((H,), dtype=np.float32)
                    ch[:take] = acts[t:t+take]
                    mk[:take] = 1.0
                    action_chunks.append(ch)
                    masks.append(mk)
                    proprios.append(proprio[t])
                    # next_state s_{t+H}
                    idx2 = min(t + H, T - 1)
                    next_props.append(proprio[idx2])
                    # reward: 1 if chunk reaches terminal (t+H >= T)
                    rc = np.zeros((H,), dtype=np.float32)
                    if t + H >= T:
                        rc[take - 1] = 1.0
                    rew_chunks.append(rc)
                    dones.append(1.0 if t + H >= T else 0.0)
                frame_counter += T
                episode_ends.append(frame_counter - 1)
        print(f"[{fi+1}/{len(files)}] {os.path.basename(path)}: cumulative {frame_counter} frames")

    proprios = np.stack(proprios, axis=0)
    action_chunks = np.stack(action_chunks, axis=0)
    masks = np.stack(masks, axis=0)
    next_props = np.stack(next_props, axis=0)
    rew_chunks = np.stack(rew_chunks, axis=0)
    dones = np.asarray(dones, dtype=np.float32)
    ends = np.asarray(episode_ends, dtype=np.int64)

    np.savez(os.path.join(args.out_dir, f'chunk{H}.npz'),
             obs=proprios, action_chunks=action_chunks, masks=masks,
             next_obs=next_props, reward_chunks=rew_chunks, done=dones,
             episode_ends=ends)
    print(f"Saved: {proprios.shape[0]} samples, chunk={H}, at "
          f"{args.out_dir}/chunk{H}.npz")


if __name__ == '__main__':
    main()
