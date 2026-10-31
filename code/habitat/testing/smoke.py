import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "training"))

from train_tea_habitat import make_env, flatten_obs, load_pretrained_policy


def main():
    cfgp = os.path.abspath(
        "habitat-lab/habitat-baselines/habitat_baselines/config/"
        "social_nav/social_nav.yaml"
    )
    env, cfg = make_env(cfgp)
    obs = env.reset()
    if isinstance(obs, list):
        obs = obs[0]
    print("OBS KEYS:", sorted(obs.keys()), flush=True)
    s = flatten_obs(obs)
    print("FLATTEN DIM:", s.shape, flush=True)
    import numpy as np
    import torch
    from train_tea_habitat import policy_act, policy_specs

    pol = load_pretrained_policy("ckpt/social_nav_latest.pth", env, cfg,
                                 device="cuda")
    pol.eval()
    n_layers, hidden, act_dim = policy_specs(pol)
    print("POLICY OK layers/hidden/act:", n_layers, hidden, act_dim,
          flush=True)
    rnn = torch.zeros(1, n_layers, hidden, device="cuda")
    prev = torch.zeros(1, act_dim, device="cuda")
    masks = torch.zeros(1, 1, dtype=torch.bool, device="cuda")
    for t in range(3):
        ob = {k: torch.as_tensor(np.asarray(v), device="cuda").unsqueeze(0)
              for k, v in obs.items() if isinstance(v, (np.ndarray, list))}
        a, rnn = policy_act(pol, ob, rnn, prev, masks, deterministic=True)
        a_np = a.squeeze(0).cpu().numpy()
        out = env.step(__import__("train_tea_habitat").build_joint_action(a_np))
        obs, r, done, info = out[0] if isinstance(out, list) else out
        if isinstance(obs, list):
            obs = obs[0]
        prev = a
        masks = torch.ones(1, 1, dtype=torch.bool, device="cuda")
        print(f"STEP {t}: action={a_np.round(3).tolist()} r={float(np.asarray(r).reshape(-1)[0]):.3f}",
              flush=True)
    print("E2E OK", flush=True)


if __name__ == "__main__":
    main()
