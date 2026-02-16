from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, Tuple
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    replay_size: int = 100000
    min_replay: int = 5000
    target_update: int = 1000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50000

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, int(a), float(r), s2, bool(done)))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buf)

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, cfg: DQNConfig, device: str = "cpu"):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.cfg = cfg
        self.device = torch.device(device)

        self.q = MLP(obs_dim, n_actions).to(self.device)
        self.tgt = MLP(obs_dim, n_actions).to(self.device)
        self.tgt.load_state_dict(self.q.state_dict())
        self.tgt.eval()

        self.opt = optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.replay = ReplayBuffer(cfg.replay_size)
        self.steps = 0

    def epsilon(self) -> float:
        frac = min(1.0, self.steps / float(self.cfg.epsilon_decay_steps))
        return self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    def act(self, obs: np.ndarray) -> int:
        self.steps += 1
        if random.random() < self.epsilon():
            return random.randrange(self.n_actions)
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.q(x)
            return int(torch.argmax(q, dim=1).item())

    def update(self) -> float:
        if len(self.replay) < self.cfg.min_replay:
            return 0.0

        s, a, r, s2, d = self.replay.sample(self.cfg.batch_size)

        s_t = torch.tensor(s, dtype=torch.float32, device=self.device)
        a_t = torch.tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)
        r_t = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        s2_t = torch.tensor(s2, dtype=torch.float32, device=self.device)
        d_t = torch.tensor(d.astype(np.float32), dtype=torch.float32, device=self.device).unsqueeze(1)

        q_sa = self.q(s_t).gather(1, a_t)
        with torch.no_grad():
            max_q_next = self.tgt(s2_t).max(dim=1, keepdim=True)[0]
            target = r_t + self.cfg.gamma * (1.0 - d_t) * max_q_next

        loss = torch.mean((q_sa - target) ** 2)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()

        if self.steps % self.cfg.target_update == 0:
            self.tgt.load_state_dict(self.q.state_dict())
        return float(loss.item())

    def save(self, path: str) -> None:
        torch.save({"q": self.q.state_dict(), "cfg": self.cfg.__dict__}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.q.load_state_dict(ckpt["q"])
        self.tgt.load_state_dict(ckpt["q"])
