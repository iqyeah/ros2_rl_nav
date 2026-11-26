import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
    VecNormalize,
)

from rl_nav.nav_env import NavEnv
from rl_nav.scenarios import make_eval_set

ROOT = Path(__file__).resolve().parent


def make_env(rank, seed):
    def _init():
        return NavEnv(seed=seed + rank * 1000)

    return _init


def main():
    p = argparse.ArgumentParser(description="Train PPO navigation policy")
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--dummy", action="store_true", help="no subprocess workers")
    p.add_argument("--run-name", default="ppo_nav")
    args = p.parse_args()

    run_dir = ROOT / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.dummy:
        venv = DummyVecEnv([make_env(i, args.seed) for i in range(args.n_envs)])
    else:
        venv = SubprocVecEnv(
            [make_env(i, args.seed) for i in range(args.n_envs)], start_method="spawn"
        )
    venv = VecMonitor(venv)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

    eval_scenarios = make_eval_set()
    eval_env = DummyVecEnv([lambda: NavEnv(seed=777, scenario_pool=eval_scenarios)])
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best"),
        eval_freq=5_000,
        n_eval_episodes=10,
        deterministic=True,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=5_000, save_path=str(run_dir / "ckpt"), name_prefix="ppo"
    )

    model = PPO(
        "MlpPolicy",
        venv,
        policy_kwargs=dict(net_arch=[256, 256], log_std_init=-0.5),
        learning_rate=3e-4,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        verbose=1,
        tensorboard_log=str(ROOT / "runs" / "tb"),
        device=args.device,
        seed=args.seed,
    )
    model.learn(args.steps, callback=[eval_cb, ckpt_cb])
    model.save(str(run_dir / "final"))
    venv.save(str(run_dir / "vecnormalize.pkl"))
    print(f"Done. Model and normalization stats saved to {run_dir}")


if __name__ == "__main__":
    main()
