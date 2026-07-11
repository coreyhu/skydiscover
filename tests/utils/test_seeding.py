"""Seed ownership must be explicit and reproducible."""

from __future__ import annotations

import json
import random
from types import SimpleNamespace

import numpy as np
import pytest

from skydiscover.config import Config, LLMModelConfig
from skydiscover.llm.llm_pool import LLMPool
from skydiscover.runner import Runner
from skydiscover.utils.seeding import apply_seed_contract, derive_seed, write_seed_report


def test_derived_seeds_are_stable_and_role_specific() -> None:
    assert derive_seed(17, "solution") == derive_seed(17, "solution")
    assert derive_seed(17, "solution") != derive_seed(17, "guide")
    assert derive_seed(17, "solution") != derive_seed(18, "solution")


@pytest.mark.parametrize("invalid", [-1, True, 1.5, "17"])
def test_invalid_seed_fails_before_rng_mutation(invalid: object) -> None:
    with pytest.raises(ValueError, match="random_seed"):
        derive_seed(invalid, "solution")


def test_process_seed_contract_reproduces_python_and_numpy() -> None:
    config = SimpleNamespace(random_seed=None)

    first_report = apply_seed_contract(23, config)
    first_values = (random.random(), float(np.random.random()))
    second_report = apply_seed_contract(23, config)
    second_values = (random.random(), float(np.random.random()))

    assert first_values == second_values
    assert first_report == second_report
    assert first_report["determinism_class"] == "seeded-best-effort"
    assert config.random_seed == first_report["applied"]["database_config"]


def test_llm_pool_selection_is_reproducible_without_coupling_roles() -> None:
    models = [
        LLMModelConfig(name="first", weight=1, init_client=lambda _: "first"),
        LLMModelConfig(name="second", weight=1, init_client=lambda _: "second"),
    ]
    first = LLMPool(models, random_seed=derive_seed(31, "solution"))
    replay = LLMPool(models, random_seed=derive_seed(31, "solution"))
    guide = LLMPool(models, random_seed=derive_seed(31, "guide"))

    first_sequence = [first._sample_model() for _ in range(20)]
    replay_sequence = [replay._sample_model() for _ in range(20)]
    guide_sequence = [guide._sample_model() for _ in range(20)]

    assert first_sequence == replay_sequence
    assert first_sequence != guide_sequence


def test_seed_report_is_atomic_and_records_unsupported_provider_seed(tmp_path) -> None:
    report = apply_seed_contract(41, SimpleNamespace())

    destination = write_seed_report(tmp_path, report)

    assert json.loads(destination.read_text(encoding="utf-8")) == report
    assert "provider_generation" in report["unsupported"]


def test_runner_publishes_the_effective_seed_report(tmp_path) -> None:
    initial_program = tmp_path / "initial.py"
    initial_program.write_text("def solve():\n    return 0\n", encoding="utf-8")
    output_dir = tmp_path / "run"

    runner = Runner(
        evaluation_file=str(tmp_path / "evaluator.py"),
        initial_program_path=str(initial_program),
        config=Config.from_dict({"random_seed": 53}),
        output_dir=str(output_dir),
    )

    persisted = json.loads((output_dir / "seed-report.json").read_text(encoding="utf-8"))
    assert persisted == runner.seed_report
    assert persisted["requested_seed"] == 53
