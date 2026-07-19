#!/usr/bin/env python
"""1K-sample smoke test — MUST pass before burning A100 hours.

Checks: data loads, shapes are right, loss decreases over ~50 steps on both
loss types, checkpoint save/resume round-trips.

TODO(week 1, after implementing the model/losses/trainer):
1. Build tiny datasets (limit=1000), batch 64, epochs 2, wandb disabled.
2. Run both loss types for 50 steps; assert final loss < initial loss.
3. Kill and re-create the Trainer; assert global_step resumed correctly.
Run on the RTX 3050 — if it fits there at batch 64, A100 runs are safe.
"""
raise SystemExit("Implement after week-1 modules are done — see docstring.")
