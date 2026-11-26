"""Watch the trained policy drive in a live window (or record GIFs headless)."""
import argparse
from pathlib import Path

from rl_nav import render
from rl_nav.nav_env import NavEnv
from rl_nav.policy import load_policy
from rl_nav.scenarios import DATA_DIR, load_eval_set

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description="Play trained policy with live rendering")
    p.add_argument("--run-dir", default=str(ROOT / "runs" / "ppo_nav"))
    p.add_argument("--model", default="best_model")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--no-show", action="store_true", help="headless, only save GIFs")
    p.add_argument("--fps", type=float, default=12.0)
    args = p.parse_args()

    import matplotlib

    if args.no_show:
        matplotlib.use("Agg")
    else:
        try:
            matplotlib.use("TkAgg")
        except ValueError:
            print("TkAgg unavailable, falling back to GIF-only mode")
            matplotlib.use("Agg")
            args.no_show = True
    import matplotlib.pyplot as plt

    show = not args.no_show
    episodes = load_eval_set()
    env = NavEnv(seed=777, scenario_pool=episodes)
    act = load_policy(args.run_dir, args.model)

    if show:
        plt.ion()
        fig, ax = plt.subplots(figsize=(7, 7))
    DATA_DIR.mkdir(exist_ok=True)

    for ep in range(args.episodes):
        obs, _ = env.reset()
        frames = []
        done = False
        outcome = "timeout"
        while not done:
            a = act(obs)
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
            outcome = info.get("outcome", "timeout")
            if show:
                render.draw_scene(ax, env.world, env.goal, env.pose, env.traj, env.scan(),
                                  title=f"ep {ep}  steps {env.steps}  last: {outcome}")
                plt.pause(1.0 / args.fps)
            else:
                frames.append(env.render())
        print(f"episode {ep}: {outcome} after {env.steps} steps")
        if not show:
            render.save_gif(frames, DATA_DIR / f"play_ep{ep}.gif")
            print(f"  saved data/play_ep{ep}.gif")
        elif outcome == "success":
            plt.pause(0.6)

    if show:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
