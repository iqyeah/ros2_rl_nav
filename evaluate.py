"""Batch-evaluate a trained policy on the fixed, unseen scenario set."""
import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rl_nav import render
from rl_nav.nav_env import NavEnv
from rl_nav.policy import load_policy
from rl_nav.scenarios import DATA_DIR, load_eval_set

ROOT = Path(__file__).resolve().parent


def run_episode(act, env, record_frames=False):
    obs, _ = env.reset()
    frames = [] if record_frames else None
    total = 0.0
    while True:
        a = act(obs)
        obs, r, term, trunc, info = env.step(a)
        total += r
        if record_frames:
            frames.append(env.render())
        if term or trunc:
            return info.get("outcome", "timeout"), total, env.steps, env.min_clearance, frames


def main():
    p = argparse.ArgumentParser(description="Evaluate trained policy on unseen maps")
    p.add_argument("--run-dir", default=str(ROOT / "runs" / "ppo_nav"))
    p.add_argument("--model", default="best_model")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=777, help="heading noise seed")
    p.add_argument("--gif", type=int, default=3, metavar="N", help="record GIFs of first N runs")
    args = p.parse_args()

    episodes = load_eval_set()[: args.n]
    env = NavEnv(seed=args.seed, scenario_pool=episodes)
    act = load_policy(args.run_dir, args.model)

    outcomes, results = [], []
    for i in range(len(episodes)):
        outcome, total, steps, min_cl, _ = run_episode(act, env)
        outcomes.append(outcome)
        results.append({"scenario": i, "outcome": outcome, "steps": steps,
                        "return": round(total, 2), "min_clearance": round(min_cl, 3)})
        print(f"[{i + 1:2d}/{len(episodes)}] {outcome:9s} steps={steps:3d} "
              f"min_clearance={min_cl:.2f}")

    counts = Counter(outcomes)
    n = len(episodes)
    succ = counts["success"]
    succ_steps = [r["steps"] for r in results if r["outcome"] == "success"]
    summary = {
        "model": args.model,
        "episodes": n,
        "success": succ,
        "collision": counts["collision"],
        "timeout": counts["timeout"],
        "success_rate": round(succ / n, 3),
        "avg_steps_to_goal": round(sum(succ_steps) / len(succ_steps), 1) if succ_steps else None,
        "avg_min_clearance": round(sum(r["min_clearance"] for r in results) / n, 3),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "eval_results.json").write_text(
        json.dumps({"summary": summary, "results": results}, indent=2)
    )

    # Trajectory overview of the first 9 scenarios
    env2 = NavEnv(seed=args.seed, scenario_pool=episodes)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    for ax, sc in zip(axes.ravel(), episodes[:9]):
        obs, _ = env2.reset()
        done = False
        while not done:
            a = act(obs)
            obs, r, term, trunc, info = env2.step(a)
            done = term or trunc
        render.draw_scene(ax, env2.world, env2.goal, env2.pose, env2.traj, None,
                          title=f"#{env2.pool_idx - 1}: {info.get('outcome')}")
    fig.tight_layout()
    fig.savefig(DATA_DIR / "eval_trajectories.png", dpi=90)
    plt.close(fig)

    if args.gif > 0:
        env3 = NavEnv(seed=args.seed, scenario_pool=episodes)
        for k in range(min(args.gif, len(episodes))):
            _, _, _, _, frames = run_episode(act, env3, record_frames=True)
            render.save_gif(frames, DATA_DIR / f"rollout_{k}.gif")
            print(f"saved data/rollout_{k}.gif")


if __name__ == "__main__":
    main()
