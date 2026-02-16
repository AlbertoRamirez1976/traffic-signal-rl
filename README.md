# Reinforcement Learning Traffic Signal Optimization (Single Intersection + Small Grid)

## Overview
This repository implements and evaluates reinforcement learning (RL) for adaptive traffic signal control in:
1) a **single intersection** scenario, and
2) a **small 2x2 grid** scenario.

The default environment is a **lightweight Python simulator** (no external simulator required), designed to be reproducible and easy to run.
Optional integration hooks are provided for SUMO/SUMO-RL if your team decides to switch to microsimulation later.

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart: Train DQN (Single Intersection)
```bash
python -m src.rl.train --scenario single --steps 60000 --seed 0
```

## Train DQN (2x2 Grid)
```bash
python -m src.rl.train --scenario grid --steps 120000 --seed 0
```

## Evaluate (compare baselines vs RL)
```bash
python -m src.eval.evaluate --scenario single --seeds 0 1 2 3 4
python -m src.eval.evaluate --scenario grid --seeds 0 1 2 3 4
```

## Demo Dashboard (Streamlit)
```bash
streamlit run dashboard.py
```

## Optional: SUMO / SUMO-RL
If you later decide to use SUMO, install SUMO and SUMO-RL separately and implement `src/env/sumo_wrapper.py` (stub included).
