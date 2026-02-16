from __future__ import annotations

import argparse
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch

from src.env.make_env import make_env
from src.baselines.fixed_time import FixedTimeController, FixedTimeGridController
from src.baselines.heuristic import MaxQueueController, MaxQueueGridController
from src.rl.dqn_agent import DQNAgent, DQNConfig

def run_episode(env, policy_fn):
    obs, _ = env.reset()
    done = False
    trunc = False
    total_reward = 0.0
    queues = []
    steps = 0
    while not (done or trunc):
        a = policy_fn(obs)
        obs, r, done, trunc, info = env.step(a)
        total_reward += float(r)
        if "total_queue" in info:
            queues.append(float(info["total_queue"]))
        else:
            queues.append(float(info["q_ns"] + info["q_ew"]))
        steps += 1
    q = np.array(queues, dtype=np.float32)
    return {
        "total_reward": float(total_reward),
        "avg_queue": float(np.mean(q)),
        "max_queue": float(np.max(q)),
        "steps": int(steps),
    }

def load_dqn(checkpoint_path: str, obs_dim: int, n_actions: int, device: str = "cpu"):
    cfg = DQNConfig()
    agent = DQNAgent(obs_dim=obs_dim, n_actions=n_actions, cfg=cfg, device=device)
    agent.load(checkpoint_path)
    agent.q.eval()

    def policy(obs):
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
            q = agent.q(x)
            return int(torch.argmax(q, dim=1).item())
    return policy

def main(args):
    out_dir = Path("results")
    tables = out_dir / "tables"
    plots = out_dir / "plots"
    tables.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in args.seeds:
        env = make_env(args.scenario, seed=seed)
        obs, _ = env.reset(seed=seed)
        obs_dim = int(np.prod(env.observation_space.shape))
        n_actions = int(env.action_space.n)

        if args.scenario == "single":
            fixed = FixedTimeController(cycle=args.fixed_cycle)
            heur = MaxQueueController()
            fixed.reset(); heur.reset()
            fixed_pol = lambda o: fixed.act(o)
            heur_pol = lambda o: heur.act(o)
        else:
            fixed = FixedTimeGridController(cycle=args.fixed_cycle)
            heur = MaxQueueGridController()
            fixed.reset(); heur.reset()
            fixed_pol = lambda o: fixed.act(o)
            heur_pol = lambda o: heur.act(o)

        m_fixed = run_episode(env, fixed_pol)
        m_heur = run_episode(env, heur_pol)

        ckpt = args.dqn_checkpoint or f"results/checkpoints/dqn_{args.scenario}_seed{seed}_final.pt"
        dqn_pol = load_dqn(ckpt, obs_dim=obs_dim, n_actions=n_actions, device=args.device)
        m_dqn = run_episode(env, dqn_pol)

        for name, m in [("fixed_time", m_fixed), ("max_queue", m_heur), ("dqn", m_dqn)]:
            rows.append({"scenario": args.scenario, "seed": int(seed), "policy": name, **m})

    csv_path = tables / f"eval_{args.scenario}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    policies = sorted(set(r["policy"] for r in rows))
    data = {p: [r["avg_queue"] for r in rows if r["policy"] == p] for p in policies}

    fig = plt.figure()
    plt.boxplot([data[p] for p in policies], labels=policies)
    plt.ylabel("Average Queue (lower is better)")
    plt.title(f"Policy Comparison - {args.scenario}")
    fig.tight_layout()
    fig_path = plots / f"avg_queue_{args.scenario}.png"
    plt.savefig(fig_path, dpi=200)
    plt.close(fig)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {fig_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", choices=["single", "grid"], default="single")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--fixed_cycle", type=int, default=20)
    p.add_argument("--dqn_checkpoint", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()
    main(args)
