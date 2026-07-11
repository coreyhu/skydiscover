"""Explicit, reproducible seed ownership for discovery runs."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

POOL_ROLES = ("solution", "evaluator", "guide", "evox_summary")


def derive_seed(base_seed: int | None, namespace: str) -> int | None:
    """Derive a stable 32-bit child seed without sharing RNG state."""

    if base_seed is None:
        return None
    if not isinstance(base_seed, int) or isinstance(base_seed, bool) or base_seed < 0:
        raise ValueError("random_seed must be a non-negative integer or null")
    digest = hashlib.sha256(f"skydiscover:{base_seed}:{namespace}".encode()).digest()
    return int.from_bytes(digest[:4], byteorder="big")


def apply_seed_contract(base_seed: int | None, database_config: Any) -> dict[str, Any]:
    """Apply process RNG seeds and return the exact best-effort seed report."""

    if base_seed is None:
        return {
            "schema_version": 1,
            "requested_seed": None,
            "determinism_class": None,
            "applied": {},
            "unsupported": {"provider_generation": "no provider-side seed is passed to model APIs"},
        }

    # Validate before mutating any RNG state.
    derive_seed(base_seed, "validation")
    numpy_seed = derive_seed(base_seed, "numpy")
    database_seed = derive_seed(base_seed, "search_database")
    random.seed(base_seed)
    np.random.seed(numpy_seed)
    database_accepts_seed = hasattr(database_config, "random_seed")
    if database_accepts_seed:
        database_config.random_seed = database_seed

    nested_seed = derive_seed(base_seed, "evox_strategy")
    return {
        "schema_version": 1,
        "requested_seed": base_seed,
        "determinism_class": "seeded-best-effort",
        "applied": {
            "python_random": base_seed,
            "numpy_random": numpy_seed,
            "database_config": database_seed if database_accepts_seed else None,
            "llm_pools": {role: derive_seed(base_seed, role) for role in POOL_ROLES},
            "evox_strategy": {
                "base": nested_seed,
                "llm_pools": {role: derive_seed(nested_seed, role) for role in POOL_ROLES},
            },
        },
        "unsupported": {
            "provider_generation": "no provider-side seed is passed to model APIs",
            "parallel_scheduling": "task interleaving is not deterministic when concurrency exceeds one",
        },
    }


def write_seed_report(output_dir: str | Path, report: dict[str, Any]) -> Path:
    """Atomically write the run's seed ownership report."""

    destination = Path(output_dir).resolve() / "seed-report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return destination
