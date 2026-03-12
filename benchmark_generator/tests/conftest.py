import json
import shutil
import textwrap
from pathlib import Path

import pytest


@pytest.fixture
def tmp_task_dir(tmp_path):
    task_dir = tmp_path / "task_001"
    task_dir.mkdir()
    return task_dir


@pytest.fixture
def sample_prompt_md():
    return textwrap.dedent("""\
        Complete the following task based on Python:
        Task requirements: Train a classifier on the iris dataset.

        Steps:
        (1) Load the iris dataset (output: "data_analysis.txt")
        (2) Train a classifier (output: "model.pkl")
        (3) Evaluate the model (output: "evaluation_results.csv")
    """)


@pytest.fixture
def sample_prompt_json():
    return {
        "prompt": "Train a classifier on iris dataset using sklearn.",
        "data_source_type": "2=open source data",
    }


@pytest.fixture
def sample_metrics_py():
    return textwrap.dedent('''\
        """Auto-generated metrics. Run: python metrics.py"""

        import os
        import sys
        import traceback


        METRICS = [
            {
                "task_name": "Predictive modeling",
                "function": "Model Accuracy",
                "metric": "Model Accuracy",
                "ground_truth": "prediction_output.npy",
                "code": (
                    "def model_accuracy(ground_truth):\\n"
                    "    import numpy as np\\n"
                    "    y_pred = np.load('prediction_output.npy')\\n"
                    "    y_gt = np.load(ground_truth)\\n"
                    "    return bool(np.mean(y_pred == y_gt) > 0.9)\\n"
                ),
            },
            {
                "task_name": "Data exploration",
                "function": "File Existence",
                "metric": "Report Exists",
                "ground_truth": None,
                "code": (
                    "def report_exists(ground_truth):\\n"
                    "    import os\\n"
                    "    return os.path.exists('data_analysis.txt')\\n"
                ),
            },
        ]


        def run_all():
            gt_dir = os.path.join(os.path.dirname(__file__), "gt")
            passed = 0
            failed = 0
            for entry in METRICS:
                name = entry["metric"]
                gt_file = entry.get("ground_truth")
                gt_path = os.path.join(gt_dir, gt_file) if gt_file else None
                code = entry["code"]
                try:
                    local_ns = {}
                    exec(code, {}, local_ns)
                    func = list(local_ns.values())[0]
                    result = func(gt_path) if gt_path else func()
                    if result:
                        print(f"  PASS: {name}")
                        passed += 1
                    else:
                        print(f"  FAIL: {name}")
                        failed += 1
                except Exception:
                    print(f"  ERROR: {name}")
                    traceback.print_exc()
                    failed += 1
            print(f"\\nResults: {passed} passed, {failed} failed")
            return failed == 0


        if __name__ == "__main__":
            ok = run_all()
            sys.exit(0 if ok else 1)
    ''')


@pytest.fixture
def sample_metric_yaml():
    return {
        "TMC-list": [
            {
                "code": (
                    "def model_accuracy(ground_truth):\n"
                    "    from sklearn.metrics import accuracy_score\n"
                    "    import numpy as np\n"
                    "    y_pred = np.load('prediction_output.npy')\n"
                    "    y = np.load(ground_truth)\n"
                    "    return accuracy_score(y, y_pred) > 0.9"
                ),
                "function": "Model Accuracy",
                "metric": "Model Accuracy",
                "task_name": "Predictive modeling",
                "ground_truth": "prediction_output.npy",
            }
        ],
        "prompt": "Train a classifier on iris dataset.",
    }


@pytest.fixture
def populated_task_dir(tmp_task_dir, sample_prompt_md, sample_metrics_py):
    (tmp_task_dir / "prompt.md").write_text(sample_prompt_md)
    (tmp_task_dir / "metrics.py").write_text(sample_metrics_py)

    gt_dir = tmp_task_dir / "gt"
    gt_dir.mkdir()
    (gt_dir / "prediction_output.npy").write_bytes(b"fake")
    (gt_dir / "data_analysis.txt").write_text("analysis")

    (tmp_task_dir / "train.csv").write_text("a,b\n1,2")

    return tmp_task_dir
