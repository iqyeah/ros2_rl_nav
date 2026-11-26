"""Load a trained policy together with its observation-normalization stats."""
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .nav_env import NavEnv


def load_policy(run_dir, model_name="best_model"):
    """Return act(obs) -> action, applying saved VecNormalize obs statistics."""
    run_dir = Path(run_dir)
    model_path = run_dir / f"{model_name}.zip"
    if not model_path.exists():
        model_path = run_dir / "final.zip"
    model = PPO.load(model_path, device="cpu")

    dummy = DummyVecEnv([lambda: NavEnv()])
    stats_path = run_dir / "vecnormalize.pkl"
    if stats_path.exists():
        vn = VecNormalize.load(str(stats_path), venv=dummy)
        mean, var, clip = vn.obs_rms.mean, vn.obs_rms.var, vn.clip_obs
    else:
        mean, var, clip = 0.0, 1.0, np.inf

    def act(obs):
        o = np.clip((obs - mean) / np.sqrt(var + 1e-8), -clip, clip)
        a, _ = model.predict(o[None, :].astype(np.float32), deterministic=True)
        return a[0]

    return act
