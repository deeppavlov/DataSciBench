import json
from pathlib import Path

import yaml

from benchmark_generator.pack_task import (
    _parse_metrics_py,
    _prompt_md_to_json,
    _metrics_to_yaml,
    pack_single_task,
)


class TestPromptMdToJson:
    def test_basic_conversion(self, sample_prompt_md):
        from io import StringIO
        from pathlib import Path
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(sample_prompt_md)
            f.flush()
            result = _prompt_md_to_json(Path(f.name))

        assert "prompt" in result
        assert result["data_source_type"] == "4=auto generated"
        assert "Train a classifier" in result["prompt"]

    def test_prompt_preserves_newlines(self, tmp_path):
        text = "Line 1\n\nLine 2\n  indented\n"
        p = tmp_path / "prompt.md"
        p.write_text(text)
        result = _prompt_md_to_json(p)
        assert result["prompt"] == text


class TestParseMetricsPy:
    def test_parse_sample(self, tmp_path, sample_metrics_py):
        p = tmp_path / "metrics.py"
        p.write_text(sample_metrics_py)
        metrics = _parse_metrics_py(p)
        assert len(metrics) == 2
        assert metrics[0]["task_name"] == "Predictive modeling"
        assert metrics[0]["function"] == "Model Accuracy"
        assert metrics[0]["ground_truth"] == "prediction_output.npy"
        assert "def model_accuracy" in metrics[0]["code"]
        assert metrics[1]["ground_truth"] is None

    def test_parse_has_code_field(self, tmp_path, sample_metrics_py):
        p = tmp_path / "metrics.py"
        p.write_text(sample_metrics_py)
        metrics = _parse_metrics_py(p)
        for m in metrics:
            assert "code" in m
            assert "def " in m["code"]


class TestMetricsToYaml:
    def test_conversion(self):
        metrics = [
            {
                "task_name": "Modeling",
                "function": "Accuracy",
                "metric": "Model Accuracy",
                "ground_truth": "pred.npy",
                "code": "def acc(gt):\n    return True",
            }
        ]
        result = _metrics_to_yaml(metrics, "Test prompt")
        assert "TMC-list" in result
        assert len(result["TMC-list"]) == 1
        assert result["TMC-list"][0]["task_name"] == "Modeling"
        assert result["prompt"] == "Test prompt"

    def test_none_gt_excluded(self):
        metrics = [
            {
                "task_name": "Report",
                "function": "Exists",
                "metric": "File Check",
                "ground_truth": None,
                "code": "def check(): return True",
            }
        ]
        result = _metrics_to_yaml(metrics, "Prompt")
        assert "ground_truth" not in result["TMC-list"][0]


class TestPackSingleTask:
    def test_full_pack(self, populated_task_dir, tmp_path):
        benchmark_root = tmp_path / "bench"
        benchmark_root.mkdir()

        pack_single_task(populated_task_dir, "test_001", benchmark_root)

        data_dir = benchmark_root / "data" / "test_001"
        metric_dir = benchmark_root / "metric" / "test_001"

        assert (data_dir / "prompt.json").exists()
        prompt = json.loads((data_dir / "prompt.json").read_text())
        assert "prompt" in prompt
        assert prompt["data_source_type"] == "4=auto generated"

        assert (data_dir / "gt").is_dir()
        assert (data_dir / "gt" / "prediction_output.npy").exists()

        assert (data_dir / "train.csv").exists()

        assert (metric_dir / "metric.yaml").exists()
        metric = yaml.safe_load((metric_dir / "metric.yaml").read_text())
        assert "TMC-list" in metric
        assert "prompt" in metric
        assert len(metric["TMC-list"]) == 2

    def test_missing_prompt_raises(self, tmp_task_dir, tmp_path):
        import pytest
        with pytest.raises(FileNotFoundError, match="prompt.md"):
            pack_single_task(tmp_task_dir, "bad", tmp_path)
