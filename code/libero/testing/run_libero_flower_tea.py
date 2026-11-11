#!/usr/bin/env python3
"""LIBERO FLOWER eval + TEA K-sample Q-rerank.

FLOWER (flow-matching VLA, flower_libero checkpoints) is the frozen teacher.
Paper setting (Stage 3(B)): K=4 stochastic chunks plus the fixed-noise anchor
chunk as an extra candidate (k >= 0), teacher-anchored Soft Best-of-N with
alpha=2.0, sigma=0.2; Q and the distance use the first action of each chunk.

Open TODO:
  - FLOWER class instantiation: we pass the flat `model:` subtree of config.yaml as
    kwargs (stripping `_target_` and `_recursive_`). If upstream FLOWER adds required
    hydra-only fields (e.g. `optimizer: DictConfig`), add a fallback that synthesizes
    a minimal OmegaConf DictConfig at runtime.
  - action chunk scoring: Q-net was trained with chunk_size=1 for LIBERO
    (ActionChunkQNetwork via train_tea_libero.py --chunk_size 1). FLOWER outputs chunk_size=10. We
    score the FIRST step of each chunk since env consumes chunks one-at-a-time
    via `multistep` counter. If a chunk-level Q-net is later trained, switch to
    full-chunk scoring (see run_libero_per_ep.py qchunk mode).
  - executes action chunk multistep-open-loop (as FLOWER is trained to do:
    `multistep: 10`). Rerank decision made once per chunk; within-chunk steps
    execute cached actions. Matches FLOWER's intended operation.
"""

import argparse
import os
import sys
import json
import time
import math
import copy
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Path setup (match run_libero_openvla_tea.py layout).
sys.path.insert(0, '/workspace/envs/libero')
sys.path.insert(0, '/workspace/envs/flower')
sys.path.insert(0, '/workspace/algorithm')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                os.pardir, os.pardir, "calvin", "training"))

DEVICE = "cuda:0"


# --------------------------- LIBERO helpers -------------------------------

def get_libero_env(task, resolution=256):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"),
                                  task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file,
                "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    return env, task_description


def get_libero_image(obs, resize_size=128):
    """Return agentview_image as [H, W, 3] uint8. FLOWER uses native LIBERO
    orientation (no 180 rotation); resize is handled by preprocess_rgb_for_flower.
    Mirror environments/flower/flower/rollout/libero_rollout.py:399.
    """
    return np.asarray(obs["agentview_image"])


def get_libero_gripper_image(obs, resize_size=128):
    """Return robot0_eye_in_hand_image as native [H, W, 3] uint8."""
    return np.asarray(obs["robot0_eye_in_hand_image"])


def quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def normalize_gripper_action(action, binarize=True):
    action[..., -1] = 2 * action[..., -1] - 1
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
    return action


def invert_gripper_action(action):
    action[..., -1] = -action[..., -1]
    return action


def state_from_obs(obs):
    ee_pos = obs["robot0_eef_pos"]       # 3
    ee_quat = obs["robot0_eef_quat"]     # 4
    ee_ori = quat2axisangle(ee_quat)     # 3
    grip = obs["robot0_gripper_qpos"]    # 2
    joints = obs["robot0_joint_pos"]     # 7
    return np.concatenate([ee_pos, ee_ori, grip, joints]).astype(np.float32)


# --------------------------- FLOWER preprocessing -------------------------

# CLIP-style normalization used in FLOWER config.yaml transforms (val).
_IMG_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_IMG_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def preprocess_rgb_for_flower(img_hwc_uint8, resize=112):
    """Convert HWC uint8 to [1, 1, 3, resize, resize] float tensor.

    Mirrors the val-time transforms in config.yaml:
      Resize(112) -> ScaleImageTensor (/255.0) -> Normalize(CLIP mean/std).

    Output shape: [B=1, T=1, C=3, H, W] as FLOWER's encode_observations expects.
    """
    pil = Image.fromarray(img_hwc_uint8).resize((resize, resize), Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0  # HWC, [0, 1]
    arr = (arr - _IMG_MEAN) / _IMG_STD               # CLIP normalize
    arr = np.transpose(arr, (2, 0, 1))               # CHW
    t = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)  # [1, 1, 3, H, W]
    return t


def build_flower_batch(agent_img, gripper_img, lang_text):
    """Assemble the dict FLOWER expects from `encode_observations`."""
    rgb_static = preprocess_rgb_for_flower(agent_img)
    rgb_gripper = preprocess_rgb_for_flower(gripper_img)
    batch = {
        "rgb_obs": {
            "rgb_static": rgb_static.to(DEVICE),
            "rgb_gripper": rgb_gripper.to(DEVICE),
        },
        "lang_text": [lang_text],
    }
    return batch


# --------------------------- FLOWER loading -------------------------------

def load_flower(ckpt_path, config_path):
    """Instantiate FLOWERVLA from top-level `model:` block in config.yaml."""
    import yaml
    from flower.models.flower import FLOWERVLA

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    model_cfg = dict(cfg.get("model", {}))
    model_cfg.pop("_target_", None)
    model_cfg.pop("_recursive_", None)
    # Resolve Hydra-style variables in config that have no value mapping.
    # `multistep: ${multistep}` -> top-level `multistep` = `${act_seq_len}`
    # -> `act_seq_len: 10`. Collapse chain manually.
    def _resolve(v):
        if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
            key = v[2:-1]
            return cfg.get(key, v)
        return v

    for k, v in list(model_cfg.items()):
        rv = _resolve(v)
        if isinstance(rv, str) and rv.startswith("${"):
            rv = _resolve(rv)
        model_cfg[k] = rv

    # FLOWER expects `multistep` as int (chunk length). Config says
    # multistep: ${multistep} -> multistep: ${act_seq_len} -> 10.
    if isinstance(model_cfg.get("multistep"), str):
        model_cfg["multistep"] = int(cfg.get("act_seq_len", 10))

    # Optimizer / lr_scheduler: FLOWER stores these but only reads in
    # configure_optimizers (train-time). At eval we never call it, so pass
    # as OmegaConf DictConfig for attribute access safety.
    from omegaconf import OmegaConf
    if "optimizer" in model_cfg and isinstance(model_cfg["optimizer"], dict):
        model_cfg["optimizer"] = OmegaConf.create(model_cfg["optimizer"])
    if "lr_scheduler" in model_cfg and isinstance(model_cfg["lr_scheduler"], dict):
        model_cfg["lr_scheduler"] = OmegaConf.create(model_cfg["lr_scheduler"])

    # Disable hydra's `load_pretrained` during construction; we load manually.
    model_cfg["load_pretrained"] = False
    model_cfg.pop("pretrained_model_path", None)

    print(f"[FLOWER] Instantiating FLOWERVLA with keys: {sorted(model_cfg.keys())}",
          flush=True)
    model = FLOWERVLA(**model_cfg)
    model._load_pretrained_weights(ckpt_path)
    model = model.to(DEVICE)
    model.eval()
    return model


# --------------------------- Q-rerank helpers ------------------------------

def q_score(q_net, state_15, action_7):
    """Score a single 7-dim action at chunk_size=1 (matches LIBERO Q-net)."""
    s = torch.tensor(state_15, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    a = torch.tensor(action_7, dtype=torch.float32, device=DEVICE).view(1, 1, 7)
    with torch.no_grad():
        q = q_net(s, a).item()
    return q


def soft_bon_select(cands, q_vals, a_greedy, alpha, sigma):
    """Soft Best-of-N rerank (Verdun 2025 arxiv:2505.03156 + BoNBoN 2406.00832).

    Picks argmax_k [log pi_proxy(a_k) + Q(a_k) / alpha].
    log pi_proxy(a_k) = -||a_k - a_greedy||^2 / (2 * sigma^2).
    alpha -> inf recovers hard argmax Q. alpha -> 0 recovers teacher argmax.
    """
    q = np.asarray(q_vals, dtype=np.float64)
    diffs = np.stack([a - a_greedy for a in cands], axis=0)
    log_pi = -0.5 * np.sum(diffs ** 2, axis=-1) / (sigma ** 2)
    if math.isinf(alpha) or alpha <= 0:
        scores = q
    else:
        scores = log_pi + q / alpha
    return int(np.argmax(scores))


# --------------------------- FLOWER sampling -------------------------------

def flower_sample_chunks(model, batch, K, seed_base=None):
    """Sample K action chunks from FLOWER for the same observation.

    Returns: np.ndarray [K, act_window_size, action_dim], all in [-1, 1].
    """
    with torch.no_grad():
        cond = model.encode_observations(batch)
        # Replicate cond across K: features [1, L, d], etc.
        def _repeat(x, K):
            if isinstance(x, torch.Tensor):
                rep_args = [K] + [1] * (x.dim() - 1)
                return x.repeat(*rep_args)
            return x

        cond_K = {}
        for k, v in cond.items():
            cond_K[k] = _repeat(v, K) if isinstance(v, torch.Tensor) else v

        # Independent noise per sample.
        if seed_base is not None:
            g = torch.Generator(device=DEVICE).manual_seed(int(seed_base))
            noise = torch.randn(
                K, model.act_window_size, model.action_dim,
                device=DEVICE, generator=g,
            )
        else:
            noise = torch.randn(
                K, model.act_window_size, model.action_dim,
                device=DEVICE,
            )
        dtype = next(model.parameters()).dtype
        noise = noise.to(dtype)
        chunks = model.sample_actions(noise, cond_K, inference=True)  # [K, T, A]
    return chunks.float().cpu().numpy()


def flower_greedy_chunk(model, batch, fixed_seed=42):
    """Deterministic chunk via fixed-seed noise; used as teacher anchor."""
    return flower_sample_chunks(model, batch, K=1, seed_base=fixed_seed)[0]


# --------------------------- main eval loop -------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task_suite", type=str, required=True,
                   choices=["libero_spatial", "libero_object",
                            "libero_goal", "libero_10"])
    p.add_argument("--flower_ckpt", type=str, required=True,
                   help="Path to model.safetensors")
    p.add_argument("--flower_config", type=str, default=None,
                   help="Path to config.yaml (default: next to ckpt)")
    p.add_argument("--tea_q", type=str, default=None,
                   help="Path to tea_q.pt (required unless --baseline)")
    p.add_argument("--baseline", action="store_true",
                   help="K=1 greedy FLOWER, no rerank")
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--num_trials_per_task", type=int, default=10)
    p.add_argument("--num_steps_wait", type=int, default=10)
    p.add_argument("--rerank_mode", type=str, default="soft",
                   choices=["hard", "soft"])
    p.add_argument("--alpha", type=float, default=2.0,
                   help="soft-BoN temperature (inf or <=0 => hard-argmax)")
    p.add_argument("--sigma", type=float, default=0.2,
                   help="teacher prior std (per-dim)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="results")
    p.add_argument("--task_ids", type=str, default=None,
                   help="Comma-separated task ids to run (default: all)")
    p.add_argument("--append_dir", type=str, default=None,
                   help="Append results to existing output dir instead of creating new")
    args = p.parse_args()

    if not args.baseline and args.tea_q is None:
        raise ValueError("--tea_q is required unless --baseline")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Defer heavy imports into main() so top-level stays light.
    from libero.libero import benchmark
    from train_tea_mdt import ActionChunkQNetwork

    # Output dir setup.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "baseline" if args.baseline else f"tea_K{args.K}_{args.rerank_mode}"
    if args.append_dir:
        out = args.append_dir
    else:
        out = os.path.join(args.output_dir,
                           f"libero_{args.task_suite}_flower_{tag}_{ts}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load FLOWER.
    config_path = args.flower_config or os.path.join(
        os.path.dirname(args.flower_ckpt), "config.yaml")
    print(f"Loading FLOWER from {args.flower_ckpt}", flush=True)
    flower = load_flower(args.flower_ckpt, config_path)
    multistep = int(flower.multistep)
    act_window = int(flower.act_window_size)
    action_dim = int(flower.action_dim)
    print(f"FLOWER: multistep={multistep} act_window={act_window} "
          f"action_dim={action_dim}", flush=True)

    # Load Q-net (optional).
    q_net = None
    if not args.baseline:
        print(f"Loading TEA Q-net from {args.tea_q}", flush=True)
        q_ckpt = torch.load(args.tea_q, map_location="cpu", weights_only=False)
        q_net = ActionChunkQNetwork(
            state_dim=q_ckpt["state_dim"],
            action_dim=q_ckpt["action_dim"],
            chunk_size=q_ckpt["chunk_size"],
        ).to(DEVICE)
        q_net.load_state_dict(q_ckpt["q_net"])
        q_net.eval()
        print(f"Q-net: chunk_size={q_ckpt['chunk_size']} "
              f"state_dim={q_ckpt['state_dim']} "
              f"action_dim={q_ckpt['action_dim']}", flush=True)

    # Benchmark.
    bench_dict = benchmark.get_benchmark_dict()
    suite = bench_dict[args.task_suite]()
    num_tasks = suite.get_num_tasks()
    max_steps = {"libero_spatial": 220, "libero_object": 280,
                 "libero_goal": 300, "libero_10": 520}[args.task_suite]

    task_ids_run = ([int(x) for x in args.task_ids.split(",")]
                    if args.task_ids else list(range(num_tasks)))

    # Resume support.
    per_task = {}
    total_eps = 0
    total_succ = 0
    if args.append_dir:
        partial_path = os.path.join(out, "results_partial.json")
        if os.path.exists(partial_path):
            with open(partial_path) as f:
                prev = json.load(f)
            per_task = prev.get("per_task", {})
            total_eps = prev.get("total_episodes", 0)
            total_succ = prev.get("total_successes", 0)
            print(f"Resumed: {total_succ}/{total_eps} from {partial_path}",
                  flush=True)

    t_start = time.time()

    for task_id in task_ids_run:
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task)
        print(f"\n[{task_id + 1}/{num_tasks}] {task_description}", flush=True)

        task_ep = 0
        task_succ = 0
        for ep in range(args.num_trials_per_task):
            env.reset()
            init = initial_states[ep % len(initial_states)]
            obs = env.set_init_state(init)
            flower.reset()
            t = 0
            success = False

            # Chunk execution cache: FLOWER trained with multistep=10 open-loop,
            # rerank decision taken at chunk boundaries.
            cached_chunk = None          # np.ndarray [act_window, action_dim]
            chunk_pos = multistep        # force replan on first real step

            try:
                while t < max_steps + args.num_steps_wait:
                    if t < args.num_steps_wait:
                        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                        t += 1
                        continue

                    # Replan at chunk boundary.
                    if cached_chunk is None or chunk_pos >= multistep:
                        agent_img = get_libero_image(obs, 224)
                        gripper_img = get_libero_gripper_image(obs, 224)
                        batch = build_flower_batch(
                            agent_img, gripper_img, task_description)
                        state15 = state_from_obs(obs)

                        if args.baseline:
                            chunk = flower_sample_chunks(flower, batch, K=1)[0]
                            cached_chunk = chunk
                        else:
                            # K stochastic samples + 1 greedy anchor.
                            cands = flower_sample_chunks(flower, batch,
                                                         K=args.K)
                            greedy = flower_greedy_chunk(
                                flower, batch,
                                fixed_seed=args.seed + ep * 1000 + t)
                            all_chunks = np.concatenate(
                                [cands, greedy[None, :, :]], axis=0)
                            # Score first step of each chunk (Q-net is chunk=1).
                            first_acts = [c[0] for c in all_chunks]
                            q_vals = [q_score(q_net, state15, a)
                                      for a in first_acts]
                            if args.rerank_mode == "hard" or math.isinf(
                                    args.alpha) or args.alpha <= 0:
                                best_idx = int(np.argmax(q_vals))
                            else:
                                best_idx = soft_bon_select(
                                    first_acts, q_vals, first_acts[-1],
                                    alpha=args.alpha, sigma=args.sigma)
                            cached_chunk = all_chunks[best_idx]
                        chunk_pos = 0

                    # Execute next action from chunk.
                    action = cached_chunk[chunk_pos].astype(np.float32).copy()
                    chunk_pos += 1

                    # FLOWER outputs raw LIBERO actions in [-1,1]; no gripper
                    # convention conversion (unlike OpenVLA). Mirror
                    # environments/flower/flower/rollout/libero_rollout.py:356.
                    obs, _, done, _ = env.step(action.tolist())
                    if done:
                        success = True
                        break
                    t += 1
            except Exception as e:
                print(f"ep {ep} exception: {type(e).__name__}: {e}",
                      flush=True)

            task_ep += 1
            total_eps += 1
            if success:
                task_succ += 1
                total_succ += 1

            # Per-ep JSON (crash-safe).
            per_ep_rec = {
                "task_suite": args.task_suite,
                "task_id": int(task_id),
                "task_description": task_description,
                "ep_id": int(ep),
                "success": int(success),
                "mode": "baseline" if args.baseline else f"K{args.K}_{args.rerank_mode}",
                "t": int(t),
            }
            with open(os.path.join(out, f"t{task_id}_e{ep}.json"), "w") as f:
                json.dump(per_ep_rec, f)

            elapsed = time.time() - t_start
            print(f"  ep {ep+1}/{args.num_trials_per_task}: "
                  f"{'OK' if success else 'FAIL'} | "
                  f"task_SR={task_succ}/{task_ep}={task_succ/task_ep:.1%} | "
                  f"total_SR={total_succ}/{total_eps}="
                  f"{total_succ/total_eps:.1%} | {elapsed/60:.1f}m",
                  flush=True)

            # Running partial checkpoint.
            cur_sr = task_succ / task_ep if task_ep else 0.0
            per_task_ep = dict(per_task)
            per_task_ep[task_description] = {
                "success": int(task_succ), "total": int(task_ep),
                "sr": cur_sr, "in_progress": True,
            }
            partial_ep = {
                "task_suite": args.task_suite,
                "total_sr": total_succ / total_eps if total_eps else 0.0,
                "total_successes": int(total_succ),
                "total_episodes": int(total_eps),
                "per_task": per_task_ep,
                "wall_minutes": (time.time() - t_start) / 60,
                "current_task_id": int(task_id),
                "config": vars(args),
            }
            with open(os.path.join(out, "results_partial.json"), "w") as f:
                json.dump(partial_ep, f, indent=2)

        env.close()
        per_task[task_description] = {
            "success": int(task_succ), "total": int(task_ep),
            "sr": task_succ / task_ep if task_ep else 0.0,
        }
        partial = {
            "task_suite": args.task_suite,
            "total_sr": total_succ / total_eps if total_eps else 0.0,
            "total_successes": int(total_succ),
            "total_episodes": int(total_eps),
            "per_task": per_task,
            "wall_minutes": (time.time() - t_start) / 60,
            "completed_tasks": int(task_id) + 1,
            "config": vars(args),
        }
        with open(os.path.join(out, "results_partial.json"), "w") as f:
            json.dump(partial, f, indent=2)
        print(f"  [checkpoint] task {task_id + 1}/{num_tasks} saved "
              f"total_SR={total_succ}/{total_eps}", flush=True)

    elapsed = time.time() - t_start
    total_sr = total_succ / total_eps if total_eps else 0.0
    print("\n=== FINAL ===")
    print(f"Suite: {args.task_suite}, mode: {tag}")
    print(f"Total: {total_succ}/{total_eps} = {total_sr:.1%} "
          f"({elapsed/60:.1f} min)")
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump({
            "task_suite": args.task_suite,
            "total_sr": float(total_sr),
            "total_successes": int(total_succ),
            "total_episodes": int(total_eps),
            "per_task": per_task,
            "wall_minutes": elapsed / 60,
            "config": vars(args),
        }, f, indent=2)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
