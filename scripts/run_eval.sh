#!/usr/bin/env bash
set -e
python -m src.eval.evaluate --scenario single --seeds 0 1 2 3 4
python -m src.eval.evaluate --scenario grid --seeds 0 1 2 3 4
