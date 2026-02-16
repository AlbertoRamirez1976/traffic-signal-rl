#!/usr/bin/env bash
set -e
python -m src.rl.train --scenario single --steps 60000 --seed 0
python -m src.rl.train --scenario grid --steps 120000 --seed 0
