import json
from typing import Any

# from metagpt.roles.di.data_interpreter import DataInterpreter
# from role import SciDataInterpreter
from metagpt.schema import Plan
from src.schemas.schemas import Evaluator, Metric


class CRMetric(Metric):
    # CR: int = None
    plan_list: list[Plan]

    def __init__(self, plan_list: list[Plan]) -> None:

        self.plan_list = plan_list

    def _get_metric(self) -> float:
        """Get the Completion Rate Metric"""

        return self._calculate_cr()

    def _calculate_cr(self) -> float:
        """Calculate the Completion Rate"""
        json_objects = self.read_json_from_list(self.plan_list)
        return self.calculate_completion_rate_from_json(json_objects)

    # helper func 1
    def read_json_from_list(self, plan: list[Plan]) -> Any:
        content = str(plan)
        # Remove any non-JSON content
        json_start_pos = content.find("## Current Plan")
        json_end_pos = content.find("## Current Task")
        content = content[json_start_pos + 16 : json_end_pos]
        return json.loads(content)

    # helper func 2
    def compliant_with_ground_truth(self, _task: Any) -> bool:
        return False

    # helper func 3
    def calculate_completion_rate_from_json(self, json_objects: Any) -> float:
        total_tasks = len(json_objects)
        task_scores = []
        for task in json_objects:
            if task["is_success"] and task["is_finished"]:
                if self.compliant_with_ground_truth(task):
                    task_scores.append(2)
                else:
                    task_scores.append(1)
            else:
                task_scores.append(0)
        successful_tasks = sum(task_scores)
        return successful_tasks / (total_tasks * 2)


class CREvaluator(Evaluator):
    def __init__(self) -> None:
        pass

    @classmethod
    def calcualte_cr(cls, cr_list: list[int]) -> float:
        max_scores = len(cr_list) * 2
        scores = sum(cr_list)

        return scores / max_scores

    def _evaluate(self, dag: list[Any]) -> float:
        """Evaluate the prompt on Completion Rate

        Args:
            - prompt (str): The prompt to evaluate
        """
        max_scores = 0
        scores = 0
        for node in dag:
            max_scores += len(node.correct_list) * 3
            scores += sum(node.correct_list) * 2 + len(node.correct_list) * int(node.is_success)

        # TODO: maybe no need to use Metric here
        return scores / max_scores
