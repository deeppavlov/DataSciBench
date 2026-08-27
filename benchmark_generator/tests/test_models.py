from benchmark_generator.core.models import (
    GeneratedTask,
    MetricEntry,
    MetricsList,
    Solution,
    TaskList,
)


class TestGeneratedTask:
    def test_minimal(self) -> None:
        t = GeneratedTask(prompt="Do something")
        assert t.prompt == "Do something"
        assert t.needs_input_data is False
        assert t.input_data_code is None

    def test_with_input_data(self) -> None:
        t = GeneratedTask(
            prompt="Train model",
            needs_input_data=True,
            input_data_code="import pandas as pd\npd.DataFrame().to_csv('data.csv')",
        )
        assert t.needs_input_data is True
        assert t.input_data_code is not None
        assert "pandas" in t.input_data_code


class TestTaskList:
    def test_from_json(self) -> None:
        data = {
            "tasks": [
                {"prompt": "Task 1"},
                {"prompt": "Task 2", "needs_input_data": True, "input_data_code": "pass"},
            ]
        }
        tl = TaskList.model_validate(data)
        assert len(tl.tasks) == 2
        assert tl.tasks[0].needs_input_data is False
        assert tl.tasks[1].input_data_code == "pass"


class TestSolution:
    def test_from_json(self) -> None:
        data = {
            "code": "print('hello')",
            "subtasks": [
                {
                    "name": "Data loading",
                    "description": "Load dataset",
                    "output_files": ["data.csv"],
                },
            ],
        }
        s = Solution.model_validate(data)
        assert "hello" in s.code
        assert len(s.subtasks) == 1
        assert s.subtasks[0].output_files == ["data.csv"]


class TestMetricEntry:
    def test_without_gt(self) -> None:
        m = MetricEntry(
            task_name="Report",
            function="File Exists",
            metric="Report Exists",
            code="def check(): return True",
        )
        assert m.ground_truth is None

    def test_with_gt(self) -> None:
        m = MetricEntry(
            task_name="Modeling",
            function="Accuracy",
            metric="Model Accuracy",
            ground_truth="pred.npy",
            code="def acc(gt): return True",
        )
        assert m.ground_truth == "pred.npy"


class TestMetricsList:
    def test_from_json(self) -> None:
        data = {
            "metrics": [
                {
                    "task_name": "Test",
                    "function": "Check",
                    "metric": "Metric1",
                    "ground_truth": "file.csv",
                    "code": "def check(gt): return True",
                },
            ]
        }
        ml = MetricsList.model_validate(data)
        assert len(ml.metrics) == 1
