import textwrap

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

        METRICS = []

        def metric(task_name, function, metric_name, ground_truth=None):
            def decorator(func):
                func.task_name = task_name
                func.function = function
                func.metric = metric_name
                func.ground_truth = ground_truth
                METRICS.append(func)
                return func
            return decorator

        @metric(task_name="Predictive modeling", function="Model Accuracy", metric_name="Model Accuracy", ground_truth="prediction_output.npy")
        def model_accuracy(ground_truth):
            import numpy as np
            y_pred = np.load('prediction_output.npy')
            y_gt = np.load(ground_truth)
            return bool(np.mean(y_pred == y_gt) > 0.9)

        @metric(task_name="Data exploration", function="File Existence", metric_name="Report Exists", ground_truth=None)
        def report_exists(ground_truth):
            import os
            return os.path.exists('data_analysis.txt')

        def run_all():
            gt_dir = os.path.join(os.path.dirname(__file__), "gt")
            passed = 0
            failed = 0
            for func in METRICS:
                name = func.metric
                gt_file = func.ground_truth
                gt_path = os.path.join(gt_dir, gt_file) if gt_file else None
                try:
                    result = func(gt_path)
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
