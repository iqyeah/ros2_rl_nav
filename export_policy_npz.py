"""Export the trained PPO MLP policy to a torch-free .npz for container inference.

Weights + obs normalization stats are extracted from SB3 checkpoints so the
Gazebo bridge node only needs numpy. A consistency check compares the numpy
forward pass against SB3's own predict() before saving.
"""
import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_nav.nav_env import NavEnv

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default=str(ROOT / "runs" / "ppo_nav"))
    p.add_argument("--model", default="best_model")
    p.add_argument("--out", default=str(ROOT / "gazebo_bridge" / "policy_params.npz"))
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    candidates = [run_dir / args.model, run_dir / "best" / args.model, run_dir / "final"]
    model_path = next((c for c in candidates if c.with_suffix(".zip").exists()), None)
    if model_path is None:
        raise SystemExit(f"no model found: tried {candidates}")
    model = PPO.load(str(model_path), device="cpu")  # SB3 appends .zip itself
    sd = model.policy.state_dict()

    def get(k):
        return sd[k].cpu().numpy()

    dummy = DummyVecEnv([lambda: NavEnv()])
    vn = VecNormalize.load(str(Path(args.run_dir) / "vecnormalize.pkl"), venv=dummy)
    mean, var, clip = vn.obs_rms.mean, vn.obs_rms.var, vn.clip_obs

    # Consistency check: numpy forward pass vs SB3 predict
    rng = np.random.default_rng(0)
    ok = True
    for _ in range(5):
        obs = rng.uniform(-1, 1, model.observation_space.shape).astype(np.float32)
        # VecNormalize sits outside the policy, so predict() receives the
        # ALREADY-normalized observation; mirror that here.
        x = np.clip((obs - mean) / np.sqrt(var + 1e-8), -clip, clip).astype(np.float32)
        h = np.tanh(x @ get("mlp_extractor.policy_net.0.weight").T + get("mlp_extractor.policy_net.0.bias"))
        h = np.tanh(h @ get("mlp_extractor.policy_net.2.weight").T + get("mlp_extractor.policy_net.2.bias"))
        mine = np.clip(h @ get("action_net.weight").T + get("action_net.bias"), [0.0, -1.0], [1.0, 1.0])
        ref = np.clip(model.predict(x[None], deterministic=True)[0][0], [0.0, -1.0], [1.0, 1.0])
        if not np.allclose(mine, ref, atol=1e-5):
            ok = False
            print(f"MISMATCH numpy={mine} sb3={ref}")
    assert ok, "numpy forward pass does not match SB3 predict"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        w1=get("mlp_extractor.policy_net.0.weight"),
        b1=get("mlp_extractor.policy_net.0.bias"),
        w2=get("mlp_extractor.policy_net.2.weight"),
        b2=get("mlp_extractor.policy_net.2.bias"),
        w3=get("action_net.weight"),
        b3=get("action_net.bias"),
        obs_mean=mean,
        obs_var=var,
    )
    print(f"OK saved {out} (numpy forward == SB3 predict on 5 random obs)")


if __name__ == "__main__":
    main()
