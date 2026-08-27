"""Test that _generate_metrics_py produces valid, parseable Python code
and that it round-trips correctly through _parse_metrics_py."""

import ast
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchmark_generator.core.models import MetricEntry, MetricsList
from benchmark_generator.tools.pack_task import _parse_metrics_py
from benchmark_generator.tools.solve_task import _generate_metrics_py


def _make_metrics_list(*entries: dict[str, Any]) -> MetricsList:
    return MetricsList(metrics=[MetricEntry(**e) for e in entries])


class TestGenerateMetricsPyValidity:
    """Test that generated metrics.py is valid Python."""

    def test_simple_metric_is_valid_python(self) -> None:
        metrics = _make_metrics_list(
            {
                "task_name": "Test task",
                "function": "Check",
                "metric": "Simple Check",
                "ground_truth": "output.csv",
                "code": "def check(ground_truth):\n    return True",
            }
        )
        result = _generate_metrics_py(metrics, "test prompt")
        # Must parse as valid Python
        ast.parse(result)

    def test_metric_with_quotes_in_code(self) -> None:
        metrics = _make_metrics_list(
            {
                "task_name": "Test",
                "function": "Check",
                "metric": "Quote Check",
                "ground_truth": "out.csv",
                "code": 'def check(gt):\n    import pandas as pd\n    df = pd.read_csv("output.csv")\n    return len(df) > 0',
            }
        )
        result = _generate_metrics_py(metrics, "prompt")
        ast.parse(result)

    def test_metric_with_no_ground_truth(self) -> None:
        metrics = _make_metrics_list(
            {
                "task_name": "Report",
                "function": "Exists",
                "metric": "File Check",
                "code": "def check():\n    import os\n    return os.path.exists('report.txt')",
            }
        )
        result = _generate_metrics_py(metrics, "prompt")
        ast.parse(result)

    def test_metric_with_backslash_in_code(self) -> None:
        metrics = _make_metrics_list(
            {
                "task_name": "Test",
                "function": "Check",
                "metric": "Backslash Check",
                "ground_truth": "out.csv",
                "code": "def check(gt):\n    path = 'some\\\\path'\n    return True",
            }
        )
        result = _generate_metrics_py(metrics, "prompt")
        ast.parse(result)

    def test_metric_with_fstring_in_code(self) -> None:
        metrics = _make_metrics_list(
            {
                "task_name": "Test",
                "function": "Check",
                "metric": "FString Check",
                "ground_truth": "out.csv",
                "code": 'def check(gt):\n    name = "world"\n    print(f"hello {name}")\n    return True',
            }
        )
        result = _generate_metrics_py(metrics, "prompt")
        ast.parse(result)

    def test_multiple_metrics(self) -> None:
        metrics = _make_metrics_list(
            {
                "task_name": "Task1",
                "function": "F1",
                "metric": "M1",
                "ground_truth": "a.csv",
                "code": "def f1(gt):\n    return True",
            },
            {
                "task_name": "Task2",
                "function": "F2",
                "metric": "M2",
                "ground_truth": None,
                "code": "def f2():\n    return True",
            },
            {
                "task_name": "Task3",
                "function": "F3",
                "metric": "M3",
                "ground_truth": "b.npy",
                "code": "def f3(gt):\n    import numpy as np\n    return True",
            },
        )
        result = _generate_metrics_py(metrics, "prompt")
        ast.parse(result)


class TestGenerateMetricsPyRoundTrip:
    """Test that generated code can be parsed back by _parse_metrics_py."""

    def test_roundtrip_simple(self, tmp_path: Path) -> None:
        metrics = _make_metrics_list(
            {
                "task_name": "Modeling",
                "function": "Accuracy",
                "metric": "Model Accuracy",
                "ground_truth": "pred.npy",
                "code": "def acc(gt):\n    return True",
            }
        )
        content = _generate_metrics_py(metrics, "prompt")
        p = tmp_path / "metrics.py"
        p.write_text(content)

        parsed = _parse_metrics_py(p)
        assert len(parsed) == 1
        assert parsed[0]["task_name"] == "Modeling"
        assert parsed[0]["ground_truth"] == "pred.npy"
        assert "def acc" in parsed[0]["code"]

    def test_roundtrip_multiple(self, tmp_path: Path) -> None:
        metrics = _make_metrics_list(
            {
                "task_name": "A",
                "function": "FA",
                "metric": "MA",
                "ground_truth": "a.csv",
                "code": "def fa(gt):\n    return True",
            },
            {
                "task_name": "B",
                "function": "FB",
                "metric": "MB",
                "ground_truth": None,
                "code": "def fb():\n    return True",
            },
        )
        content = _generate_metrics_py(metrics, "prompt")
        p = tmp_path / "metrics.py"
        p.write_text(content)

        parsed = _parse_metrics_py(p)
        assert len(parsed) == 2
        assert parsed[0]["task_name"] == "A"
        assert parsed[1]["ground_truth"] is None

    def test_roundtrip_with_quotes(self, tmp_path: Path) -> None:
        metrics = _make_metrics_list(
            {
                "task_name": "CSV Check",
                "function": "Shape",
                "metric": "Shape Match",
                "ground_truth": "data.csv",
                "code": 'def shape(gt):\n    import pandas as pd\n    df = pd.read_csv("output.csv")\n    gt_df = pd.read_csv(gt)\n    return df.shape == gt_df.shape',
            }
        )
        content = _generate_metrics_py(metrics, "prompt")
        p = tmp_path / "metrics.py"
        p.write_text(content)

        parsed = _parse_metrics_py(p)
        assert len(parsed) == 1
        assert "read_csv" in parsed[0]["code"]


class TestGenerateMetricsPyExecution:
    """Test that generated metrics.py can actually be executed."""

    def test_execution_basic(self, tmp_path: Path) -> None:
        metrics = _make_metrics_list(
            {
                "task_name": "File check",
                "function": "Exists",
                "metric": "File Existence",
                "ground_truth": None,
                "code": "def check():\n    return True",
            }
        )
        content = _generate_metrics_py(metrics, "prompt")
        p = tmp_path / "metrics.py"
        p.write_text(content)

        gt_dir = tmp_path / "gt"
        gt_dir.mkdir()

        result = subprocess.run(
            [sys.executable, str(p)],
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert "PASS" in result.stdout or "FAIL" in result.stdout or "ERROR" in result.stdout, (
            f"Unexpected output: stdout={result.stdout!r} stderr={result.stderr!r}"
        )
