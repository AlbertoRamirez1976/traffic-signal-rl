from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

@dataclass
class PhaseConfig:
    min_green: int = 5
    yellow: int = 2

class _PoissonArrival:
    def __init__(self, rate: float, rng: np.random.Generator):
        self.rate = rate
        self.rng = rng

    def step(self) -> int:
        return int(self.rng.poisson(self.rate))

class SingleIntersectionEnv(gym.Env):
    """Two-movement intersection: NS vs EW."""

    def __init__(self, config: Dict[str, Any], seed: int = 0):
        self.cfg = config
        self.dt = float(config.get("dt", 1.0))
        self.horizon = int(config.get("episode_horizon", 600))
        self.phase_cfg = PhaseConfig(
            min_green=int(config.get("min_green", 5)),
            yellow=int(config.get("yellow", 2)),
        )
        self.service_rate = float(config.get("service_rate", 1.2))
        self.arrival_ns = float(config.get("arrival_rate_ns", 0.35))
        self.arrival_ew = float(config.get("arrival_rate_ew", 0.25))

        self.rng = np.random.default_rng(seed)
        self.obs_cap = 200.0

        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([self.obs_cap, self.obs_cap, 1, 600], dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)

        self._arr_ns = _PoissonArrival(self.arrival_ns, self.rng)
        self._arr_ew = _PoissonArrival(self.arrival_ew, self.rng)
        self.reset(seed=seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self._arr_ns = _PoissonArrival(self.arrival_ns, self.rng)
            self._arr_ew = _PoissonArrival(self.arrival_ew, self.rng)

        self._t = 0
        self._q_ns = 0.0
        self._q_ew = 0.0
        self._phase = 0
        self._phase_elapsed = 0
        self._pending_switch = None
        self._yellow_timer = 0
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        return np.array([
            min(self._q_ns, self.obs_cap),
            min(self._q_ew, self.obs_cap),
            float(self._phase),
            float(self._phase_elapsed),
        ], dtype=np.float32)

    def step(self, action: int):
        self._t += 1

        # arrivals
        self._q_ns += self._arr_ns.step()
        self._q_ew += self._arr_ew.step()

        # switching logic with min-green and yellow
        if self._yellow_timer > 0:
            self._yellow_timer -= 1
        else:
            if action != self._phase and self._phase_elapsed >= self.phase_cfg.min_green:
                self._pending_switch = int(action)
                self._yellow_timer = self.phase_cfg.yellow
                self._phase_elapsed = 0
            elif self._pending_switch is not None and self._yellow_timer == 0:
                self._phase = int(self._pending_switch)
                self._pending_switch = None
                self._phase_elapsed = 0
            else:
                self._phase_elapsed += 1

            # service during green
            served = self.service_rate * self.dt
            if self._yellow_timer == 0:
                if self._phase == 0:
                    self._q_ns = max(0.0, self._q_ns - served)
                else:
                    self._q_ew = max(0.0, self._q_ew - served)

        reward = -float(self._q_ns + self._q_ew)
        terminated = self._t >= self.horizon
        truncated = False
        info = {"t": self._t, "q_ns": float(self._q_ns), "q_ew": float(self._q_ew), "phase": int(self._phase)}
        return self._get_obs(), reward, terminated, truncated, info

class Grid2x2Env(gym.Env):
    """2x2 grid with 4 intersections, minimal coupling."""

    def __init__(self, config: Dict[str, Any], seed: int = 0):
        self.cfg = config
        self.grid_h, self.grid_w = config.get("grid_size", [2, 2])
        self.dt = float(config.get("dt", 1.0))
        self.horizon = int(config.get("episode_horizon", 600))
        self.min_green = int(config.get("min_green", 5))
        self.yellow = int(config.get("yellow", 2))
        self.service_rate = float(config.get("service_rate", 1.0))
        self.arrival_rate = float(config.get("arrival_rate", 0.25))
        self.turn_prob = float(config.get("turn_prob", 0.15))

        self.rng = np.random.default_rng(seed)
        self.obs_cap = 200.0

        self.observation_space = spaces.Box(
            low=np.zeros(16, dtype=np.float32),
            high=np.array([self.obs_cap, self.obs_cap, 1, 600] * 4, dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(16)  # 4 bits, one per intersection
        self.reset(seed=seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._t = 0
        self.q_ns = np.zeros(4, dtype=np.float32)
        self.q_ew = np.zeros(4, dtype=np.float32)
        self.phase = np.zeros(4, dtype=np.int32)
        self.phase_elapsed = np.zeros(4, dtype=np.int32)
        self.yellow_timer = np.zeros(4, dtype=np.int32)
        return self._get_obs(), {}

    def _decode_action(self, a: int) -> np.ndarray:
        return np.array([(a >> i) & 1 for i in range(4)], dtype=np.int32)

    def _get_obs(self) -> np.ndarray:
        obs = []
        for i in range(4):
            obs.extend([
                float(min(self.q_ns[i], self.obs_cap)),
                float(min(self.q_ew[i], self.obs_cap)),
                float(self.phase[i]),
                float(self.phase_elapsed[i]),
            ])
        return np.array(obs, dtype=np.float32)

    def step(self, action: int):
        self._t += 1
        a_vec = self._decode_action(int(action))

        # exogenous arrivals
        self.q_ns += self.rng.poisson(self.arrival_rate, size=4)
        self.q_ew += self.rng.poisson(self.arrival_rate, size=4)

        served = self.service_rate * self.dt
        for i in range(4):
            if self.yellow_timer[i] > 0:
                self.yellow_timer[i] -= 1
                continue

            if a_vec[i] != self.phase[i] and self.phase_elapsed[i] >= self.min_green:
                self.yellow_timer[i] = self.yellow
                self.phase_elapsed[i] = 0
                self.phase[i] = int(a_vec[i])  # switch after yellow in a richer model; simplified here
            else:
                self.phase_elapsed[i] += 1

            if self.yellow_timer[i] == 0:
                if self.phase[i] == 0:
                    self.q_ns[i] = max(0.0, self.q_ns[i] - served)
                else:
                    self.q_ew[i] = max(0.0, self.q_ew[i] - served)

        # tiny coupling: occasional transfer to a neighbor
        def neighbor_targets(idx: int):
            targets = []
            if idx in (0, 2):
                targets.append(idx + 1)
            if idx in (0, 1):
                targets.append(idx + 2)
            return targets

        transfers = self.rng.binomial(n=1, p=self.turn_prob, size=4)
        for i in range(4):
            if transfers[i] == 0:
                continue
            targets = neighbor_targets(i)
            if not targets:
                continue
            j = int(self.rng.choice(targets))
            if self.rng.random() < 0.5:
                self.q_ns[j] += 1
            else:
                self.q_ew[j] += 1

        total_q = float(np.sum(self.q_ns + self.q_ew))
        reward = -total_q
        terminated = self._t >= self.horizon
        truncated = False
        info = {"t": self._t, "total_queue": total_q}
        return self._get_obs(), reward, terminated, truncated, info

def load_config(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def make_env(scenario: str, seed: int = 0) -> gym.Env:
    scenario = scenario.lower().strip()
    root = Path(__file__).resolve().parents[2]
    if scenario == "single":
        cfg = load_config(root / "scenarios" / "single_intersection" / "config.json")
        return SingleIntersectionEnv(cfg, seed=seed)
    if scenario == "grid":
        cfg = load_config(root / "scenarios" / "small_grid_2x2" / "config.json")
        return Grid2x2Env(cfg, seed=seed)
    raise ValueError("Scenario must be 'single' or 'grid'")
