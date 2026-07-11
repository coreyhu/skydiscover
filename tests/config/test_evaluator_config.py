"""Tests for EvaluatorConfig defaults."""

from skydiscover.config import EvaluatorConfig


class TestEvaluatorConfigDefaults:
    def test_default_timeout(self):
        assert EvaluatorConfig().timeout == 360

    def test_default_max_retries(self):
        assert EvaluatorConfig().max_retries == 3

    def test_evaluation_budget_is_opt_in(self):
        config = EvaluatorConfig()
        assert config.max_candidate_evaluations is None
        assert config.evaluation_budget_report_path is None
