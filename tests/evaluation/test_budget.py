"""Tests for hard evaluator-execution accounting."""

import json
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock

import pytest

from skydiscover.config import EvaluatorConfig
from skydiscover.evaluation.budget import EvaluationBudget, EvaluationBudgetExceeded
from skydiscover.evaluation.evaluator import Evaluator


def test_concurrent_reservations_never_exceed_cap(tmp_path):
    report_path = tmp_path / "evaluation-budget.json"
    budget = EvaluationBudget(
        max_candidate_evaluations=8,
        report_path=str(report_path),
    )

    def try_reserve():
        try:
            budget.reserve()
            return True
        except EvaluationBudgetExceeded:
            return False

    with ThreadPoolExecutor(max_workers=16) as pool:
        accepted = list(pool.map(lambda _: try_reserve(), range(64)))

    assert sum(accepted) == 8
    report = json.loads(report_path.read_text())
    assert report["status"] == "exhausted"
    assert report["termination_reason"] == "candidate_evaluation_budget_exhausted"
    assert report["scopes"]["candidate"] == {
        "limit": 8,
        "consumed": 8,
        "remaining": 0,
        "exhausted": True,
    }


def test_report_restores_counts_for_resume(tmp_path):
    report_path = tmp_path / "evaluation-budget.json"
    first = EvaluationBudget(max_candidate_evaluations=3, report_path=str(report_path))
    first.reserve("initial")
    first.reserve("candidate")
    first.reserve("candidate")

    resumed = EvaluationBudget(max_candidate_evaluations=3, report_path=str(report_path))
    resumed.reserve("candidate")

    callback = []
    resumed.set_exhaustion_callback(lambda: callback.append(True))
    assert callback == [True]
    with pytest.raises(EvaluationBudgetExceeded):
        resumed.reserve("candidate")

    scopes = resumed.snapshot()["scopes"]
    assert scopes["initial"]["consumed"] == 1
    assert scopes["candidate"]["consumed"] == 3


def test_resume_rejects_a_changed_limit(tmp_path):
    report_path = tmp_path / "evaluation-budget.json"
    budget = EvaluationBudget(max_candidate_evaluations=2, report_path=str(report_path))
    budget.reserve()

    with pytest.raises(ValueError, match="different max_candidate_evaluations"):
        EvaluationBudget(max_candidate_evaluations=3, report_path=str(report_path))


@pytest.mark.parametrize("value", [-1, True, 1.5, "3"])
def test_invalid_limits_are_rejected(value):
    with pytest.raises(ValueError, match="non-negative integer"):
        EvaluationBudget(max_candidate_evaluations=value)


@pytest.mark.asyncio
async def test_retry_attempts_each_consume_a_reservation(tmp_path, monkeypatch):
    evaluation_file = tmp_path / "evaluate.py"
    evaluation_file.write_text(
        "calls = 0\n"
        "def evaluate(program_path):\n"
        "    global calls\n"
        "    calls += 1\n"
        "    raise RuntimeError('retry me')\n"
    )
    report_path = tmp_path / "evaluation-budget.json"
    evaluator = Evaluator(
        EvaluatorConfig(
            evaluation_file=str(evaluation_file),
            cascade_evaluation=False,
            max_retries=3,
            max_candidate_evaluations=2,
            evaluation_budget_report_path=str(report_path),
        )
    )
    monkeypatch.setattr(
        "skydiscover.evaluation.evaluator.asyncio.sleep", AsyncMock()
    )

    with pytest.raises(EvaluationBudgetExceeded):
        await evaluator.evaluate_program("pass")

    assert evaluator._eval_module.calls == 2
    assert evaluator.budget.snapshot()["scopes"]["candidate"]["consumed"] == 2
    evaluator.close()


@pytest.mark.asyncio
async def test_cascade_stages_each_consume_a_reservation(tmp_path):
    evaluation_file = tmp_path / "evaluate.py"
    evaluation_file.write_text(
        "def evaluate(program_path):\n"
        "    return {'combined_score': 1.0}\n"
        "def evaluate_stage1(program_path):\n"
        "    return {'combined_score': 0.8}\n"
        "def evaluate_stage2(program_path):\n"
        "    return {'combined_score': 0.9}\n"
    )
    evaluator = Evaluator(
        EvaluatorConfig(
            evaluation_file=str(evaluation_file),
            cascade_evaluation=True,
            max_retries=0,
            max_candidate_evaluations=2,
        )
    )

    result = await evaluator.evaluate_program("pass")

    assert result.metrics["combined_score"] == 0.9
    assert evaluator.budget.snapshot()["scopes"]["candidate"]["consumed"] == 2
    evaluator.close()


@pytest.mark.asyncio
async def test_initial_and_final_test_do_not_consume_candidate_budget(tmp_path):
    evaluation_file = tmp_path / "evaluate.py"
    evaluation_file.write_text(
        "def evaluate(program_path):\n"
        "    return {'combined_score': 1.0}\n"
    )
    evaluator = Evaluator(
        EvaluatorConfig(
            evaluation_file=str(evaluation_file),
            cascade_evaluation=False,
            max_retries=0,
            max_candidate_evaluations=1,
        )
    )

    await evaluator.evaluate_program("pass", budget_scope="initial")
    await evaluator.evaluate_program("pass", mode="test", budget_scope="final_test")
    await evaluator.evaluate_program("pass")

    scopes = evaluator.budget.snapshot()["scopes"]
    assert scopes["initial"]["consumed"] == 1
    assert scopes["final_test"]["consumed"] == 1
    assert scopes["candidate"]["consumed"] == 1
    evaluator.close()


def test_zero_budget_stops_controller_when_callback_is_attached():
    budget = EvaluationBudget(max_candidate_evaluations=0)
    stopped = []
    budget.set_exhaustion_callback(lambda: stopped.append(True))

    with pytest.raises(EvaluationBudgetExceeded):
        budget.reserve()

    assert stopped == [True]
    assert budget.snapshot()["scopes"]["candidate"]["consumed"] == 0


def test_unknown_scope_cannot_bypass_candidate_cap():
    budget = EvaluationBudget(max_candidate_evaluations=1)

    with pytest.raises(ValueError, match="Unknown evaluation budget scope"):
        budget.reserve("canddiate")
