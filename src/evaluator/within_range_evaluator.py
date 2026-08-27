from typing import Any

from src.evaluator.generic_evaluator import GenericEvaluator, GenericMetric, Rule
from src.schemas.schemas import TestFunction


class WithinRangeMetric(GenericMetric):
    def __init__(self, f: TestFunction, ground_truth: Any) -> None:
        self.f = f
        self.gt = ground_truth

    def _get_metric(self, output: Any) -> Any:  # type: ignore[override]
        """Get the Generic Metric"""

        return self.f.test(self.gt, output)  # type: ignore[call-arg]


class WithinRangeRule(Rule):
    def __init__(self, rule: str) -> None:
        self.rule = eval(rule)

    def _apply_rule(self, metric: Any, **_kwargs: Any) -> Any:
        return self.rule(metric)


class WithinRangeEvaluator(GenericEvaluator):
    def __init__(self, test_func: TestFunction, ground_truth: Any, rule: str, **_kwargs: Any) -> None:
        self.f = test_func
        self.gt = ground_truth
        self.rule = WithinRangeRule(rule)

    def _evaluate(self, output: Any, **_kwargs: Any) -> Any:  # type: ignore[override]
        """Evaluate the prompt on Generic Metric

        Args:
            - prompt (Any): The prompt to evaluate
        """
        metric = WithinRangeMetric(self.f, self.gt).get_metric(output)
        return self.rule.apply_rule(metric)
