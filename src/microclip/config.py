"""YAML config loader with `_base_` inheritance and dotted CLI overrides.

Precedence (lowest → highest): base file(s) → run config → CLI overrides.
CLI override syntax: `--set train.lr=1e-4 loss.type=softmax`.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base_rel = cfg.pop("_base_", None)
    if base_rel is not None:
        base_cfg = load_config(path.parent / base_rel)
        cfg = _deep_merge(base_cfg, cfg)
    return cfg


def _parse_value(raw: str) -> Any:
    val = yaml.safe_load(raw)  # "true" -> bool, "0.5" -> float, "sgd" -> str
    if isinstance(val, str):
        # YAML 1.1 quirk: "1e-4" parses as a string (wants "1.0e-4").
        try:
            return float(val)
        except ValueError:
            return val
    return val


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply `a.b.c=value` style overrides in place and return cfg."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item!r}")
        key, raw = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ValueError(f"Cannot override through non-dict at {p!r} in {key!r}")
        node[parts[-1]] = _parse_value(raw)
    return cfg
