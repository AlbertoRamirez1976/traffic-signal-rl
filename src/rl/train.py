from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
from tqdm import trange

from src.env.make_env import make_env
from src.rl.dqn_agent import DQNAgent, DQNConfig
from src.utils.seeding import seed_everything

def run(args):
    seed_everything(args.seed)
    env = make_env(args.scenario, seed=args.seed)

    obs, _ = env.reset(seed=args.seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)

    cfg = DQNConfig(
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        min_replay=args.min_replay,
        target_update=args.target_update,
        epsilon_decay_steps=args.epsilon_decay_steps,
    )
    agent = DQNAgent(obs_dim=obs_dim, n_actions=n_actions, cfg=cfg, device=args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log = []
    ep_reward = 0.0
    ep = 0

    for step in trange(args.steps, desc="Training"):
        a = agent.act(obs)
        obs2, r, done, trunc, info = env.step(a)
        agent.replay.push(obs, a, r, obs2, done or trunc)
        loss = agent.update()
        ep_reward += float(r)
        obs = obs2

        if done or trunc:
            log.append({"episode": ep, "step": step, "episode_reward": ep_reward})
            ep += 1
            ep_reward = 0.0
            obs, _ = env.reset()

        if (step + 1) % args.save_every == 0:
            ckpt_path = out_dir / f"dqn_{args.scenario}_seed{args.seed}_step{step+1}.pt"
            agent.save(str(ckpt_path))

    ckpt_path = out_dir / f"dqn_{args.scenario}_seed{args.seed}_final.pt"
    agent.save(str(ckpt_path))

    (out_dir / f"train_log_{args.scenario}_seed{args.seed}.json").write_text(json.dumps(log, indent=2))
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", choices=["single", "grid"], default="single")
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")

    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--replay_size", type=int, default=100000)
    p.add_argument("--min_replay", type=int, default=5000)
    p.add_argument("--target_update", type=int, default=1000)
    p.add_argument("--epsilon_decay_steps", type=int, default=50000)

    p.add_argument("--save_every", type=int, default=20000)
    p.add_argument("--out_dir", type=str, default="results/checkpoints")
    args = p.parse_args()
    run(args)
