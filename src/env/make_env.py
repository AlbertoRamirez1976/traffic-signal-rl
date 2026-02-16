from __future__ import annotations
from typing import Literal
import gymnasium as gym
from .simple_traffic_env import make_env as make_simple_env

Scenario = Literal["single", "grid"]

def make_env(scenario: Scenario, seed: int = 0) -> gym.Env:
    return make_simple_env(scenario=scenario, seed=seed)
