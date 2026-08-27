from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
import yaml

if TYPE_CHECKING:
    from pathlib import Path

from scripts.validate_benchmark import main, validate

GOOD_CODE = "def check_output(ground_truth):\n    return True\n"
BROKEN_CODE = "def check_output(ground_truth):\n    return (\n"


def write_metric(root: Path, task_id: str, tmc_list: list[dict]) -> None:
    path = root / "metric" / task_id
    path.mkdir(parents=True)
    (path / "metric.yaml").write_text(yaml.dump({"prompt": "do it", "TMC-list": tmc_list}))


def write_data(root: Path, task_id: str, prompt: dict | None, gt_files: list[str]) -> None:
    path = root / "data" / task_id
    (path / "gt").mkdir(parents=True)
    if prompt is not None:
        (path / "prompt.json").write_text(json.dumps(prompt))
    for name in gt_files:
        (path / "gt" / name).write_text("value\n1\n")


def build_benchmark(root: Path) -> None:
    write_data(root, "good_0", {"prompt": "do it", "data_source_type": "1=no dependency"}, ["output.csv"])
    write_metric(
        root,
        "good_0",
        [
            {"code": GOOD_CODE, "function": "Data Completeness", "metric": "Output Exists", "task_name": "cleaning"},
            {
                "code": GOOD_CODE,
                "function": "Data Quality Score",
                "metric": "Output Matches",
                "task_name": "cleaning",
                "ground_truth": "output.csv",
            },
        ],
    )
    write_data(root, "bad_0", {"prompt": "do it"}, [])
    write_metric(
        root,
        "bad_0",
        [
            {"code": BROKEN_CODE, "function": "Data Completeness", "metric": "Broken", "task_name": "cleaning"},
            {
                "code": GOOD_CODE,
                "function": "Data Quality Score",
                "metric": "Duplicated",
                "task_name": "cleaning",
                "ground_truth": "absent.csv",
            },
            {"code": GOOD_CODE, "function": "Data Integrity", "metric": "Duplicated", "task_name": "cleaning"},
        ],
    )
    (root / "data" / "metadata").mkdir()
    (root / "data" / "human_prompt.csv").write_text("id\n1\n")


@pytest.fixture
def benchmark(tmp_path: Path) -> Path:
    build_benchmark(tmp_path)
    return tmp_path


def checks_by_task(root: Path, task_id: str = "all") -> set[tuple[str, str]]:
    _, problems = validate(str(root), task_id)
    return {(problem.task_id, problem.check) for problem in problems}


def checks(root: Path, task_id: str) -> set[str]:
    _, problems = validate(str(root), task_id)
    return {problem.check for problem in problems}


def make_task(root: Path, task_id: str, item: dict[str, Any], prompt: dict | None = None) -> None:
    write_data(root, task_id, prompt or {"prompt": "do it", "data_source_type": "1"}, ["output.csv"])
    base = {"code": GOOD_CODE, "function": "F", "metric": "M", "task_name": "t"}
    write_metric(root, task_id, [{**base, **item}])


def test_valid_task_has_no_problems(benchmark: Path) -> None:
    checked, problems = validate(str(benchmark), "good_0")
    assert checked == 1
    assert problems == []


def test_broken_task_reports_expected_checks(benchmark: Path) -> None:
    checked, problems = validate(str(benchmark), "bad_0")
    assert checked == 1
    assert {problem.check for problem in problems} == {"prompt", "gt", "code", "ground-truth", "metric-duplicate"}


def test_metadata_and_root_csv_are_not_tasks(benchmark: Path) -> None:
    checked, _ = validate(str(benchmark), "all")
    assert checked == 2
    assert not any(task == "metadata" for task, _ in checks_by_task(benchmark))


def test_layout_mismatch_reported(benchmark: Path) -> None:
    write_data(benchmark, "data_only", {"prompt": "x", "data_source_type": "1"}, ["output.csv"])
    write_metric(
        benchmark,
        "metric_only",
        [{"code": GOOD_CODE, "function": "F", "metric": "M", "task_name": "t"}],
    )
    result = checks_by_task(benchmark)
    assert ("data_only", "layout") in result
    assert ("metric_only", "layout") in result
    assert ("data_only", "metric-file") in result
    assert ("metric_only", "gt") in result


def test_unknown_task_id(benchmark: Path) -> None:
    checked, problems = validate(str(benchmark), "nope")
    assert checked == 0
    assert [problem.check for problem in problems] == ["layout"]


def test_main_exit_code_zero(benchmark: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["validate_benchmark", "--root", str(benchmark), "--task-id", "good_0"])
    assert main() == 0


def test_main_exit_code_one_and_json(
    benchmark: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["validate_benchmark", "--root", str(benchmark), "--task-id", "bad_0", "--json"])
    assert main() == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["checked"] == 1
    assert {problem["check"] for problem in payload["problems"]} == {
        "prompt",
        "gt",
        "code",
        "ground-truth",
        "metric-duplicate",
    }


def test_empty_root_is_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checked, problems = validate(str(tmp_path), "all")
    assert checked == 0
    assert {problem.check for problem in problems} == {"root"}
    monkeypatch.setattr("sys.argv", ["validate_benchmark", "--root", str(tmp_path)])
    assert main() == 1


def test_missing_root_dir_is_error(tmp_path: Path) -> None:
    checked, problems = validate(str(tmp_path / "no-such"), "all")
    assert checked == 0
    assert len(problems) == 3


def test_zero_tasks_is_error(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "metric").mkdir()
    checked, problems = validate(str(tmp_path), "all")
    assert checked == 0
    assert [problem.check for problem in problems] == ["root"]


@pytest.mark.parametrize(
    "code",
    [
        "def check(ground_truth):\n    return True\n",
        "def check(ground_truth, extra=1):\n    return True\n",
        "def check(ground_truth, *rest):\n    return True\n",
        'def check(ground_truth="x"):\n    return True\n',
        "def check(ground_truth, /):\n    return True\n",
        "def check(ground_truth, *, other=1):\n    return True\n",
        "import math\ndef check(ground_truth):\n    return True\n",
        "import numpy as np\ndef check(ground_truth):\n    return True\n",
        "import functools\n@functools.cache\ndef check(ground_truth):\n    return True\n",
        "def check(ground_truth):\n    return True\nf = lambda: 1\n",
        "def check(ground_truth):\n    return True\ndef other(ground_truth):\n    return True\n",
        "def check(ground_truth):\n    return isinstance(os.sep, str)\n",
        "def check(ground_truth):\n    import numpy as np\n    return bool(np.mean([1]))\n",
    ],
)
def test_code_accepted(tmp_path: Path, code: str) -> None:
    make_task(tmp_path, "task", {"code": code})
    assert checks(tmp_path, "task") == set()


@pytest.mark.parametrize(
    "code",
    [
        BROKEN_CODE,
        "def check():\n    return True\n",
        "def check(**kwargs):\n    return True\n",
        "def check(ground_truth, other):\n    return True\n",
        "def check(ground_truth, *, other):\n    return True\n",
        "from math import sqrt\ndef check(ground_truth):\n    return True\n",
        "class Helper:\n    pass\ndef check(ground_truth):\n    return True\n",
        "f = lambda: 1\ndef check(ground_truth):\n    return True\n",
        "check = 1\n",
        "",
    ],
)
def test_code_rejected(tmp_path: Path, code: str) -> None:
    make_task(tmp_path, "task", {"code": code})
    assert checks(tmp_path, "task") == {"code"}


@pytest.mark.parametrize(
    "code",
    [
        "THRESHOLD = 0.5\ndef check(ground_truth):\n    return THRESHOLD > 0\n",
        "import math\ndef check(ground_truth):\n    return math.sqrt(4) > 0\n",
        "def check(ground_truth):\n    return check\n",
    ],
)
def test_code_using_top_level_names_rejected(tmp_path: Path, code: str) -> None:
    make_task(tmp_path, "task", {"code": code})
    assert checks(tmp_path, "task") == {"code"}


def test_async_code_rejected(tmp_path: Path) -> None:
    make_task(tmp_path, "task", {"code": "async def check(ground_truth):\n    return False\n"})
    assert checks(tmp_path, "task") == {"code"}


def test_rejected_code_names_the_selected_object(tmp_path: Path) -> None:
    make_task(tmp_path, "task", {"code": "from math import sqrt\ndef check(ground_truth):\n    return True\n"})
    _, problems = validate(str(tmp_path), "task")
    assert len(problems) == 1
    assert "'sqrt'" in problems[0].message


def test_code_is_not_string(tmp_path: Path) -> None:
    make_task(tmp_path, "task", {"code": 42})
    assert checks(tmp_path, "task") == {"code"}


def test_code_is_not_executed(tmp_path: Path) -> None:
    marker = tmp_path / "marker.txt"
    code = f"open({str(marker)!r}, 'w').close()\ndef check(ground_truth):\n    return True\n"
    make_task(tmp_path, "task", {"code": code})
    assert checks(tmp_path, "task") == set()
    assert not marker.exists()


def test_ground_truth_accepted(tmp_path: Path) -> None:
    make_task(tmp_path, "task", {"ground_truth": "output.csv"})
    assert checks(tmp_path, "task") == set()


@pytest.mark.parametrize("value", ["../../secret.csv", "../../../secret.csv", "absent.csv"])
def test_ground_truth_rejected(tmp_path: Path, value: str) -> None:
    (tmp_path / "secret.csv").write_text("secret\n")
    make_task(tmp_path, "task", {"ground_truth": value})
    assert checks(tmp_path, "task") == {"ground-truth"}


@pytest.mark.parametrize("value", ["", "../prompt.json"])
def test_ground_truth_inside_task_dir_accepted(tmp_path: Path, value: str) -> None:
    make_task(tmp_path, "task", {"ground_truth": value})
    assert checks(tmp_path, "task") == set()


def test_ground_truth_absolute_path_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside.csv"
    outside.write_text("secret\n")
    make_task(tmp_path, "task", {"ground_truth": str(outside)})
    assert checks(tmp_path, "task") == {"ground-truth"}


def test_ground_truth_directory_rejected(tmp_path: Path) -> None:
    make_task(tmp_path, "task", {"ground_truth": "sub"})
    (tmp_path / "data" / "task" / "gt" / "sub").mkdir()
    assert checks(tmp_path, "task") == {"ground-truth"}


@pytest.mark.parametrize("value", ["CR", "", "   ", 7, None])
def test_metric_name_rejected(tmp_path: Path, value: object) -> None:
    make_task(tmp_path, "task", {"metric": value})
    assert checks(tmp_path, "task") == {"metric-name"}


def test_metric_duplicate_reported(benchmark: Path) -> None:
    assert ("bad_0", "metric-duplicate") in checks_by_task(benchmark)


def test_evaluator_metric_name_rejected(tmp_path: Path) -> None:
    make_task(tmp_path, "task", {"metric": "test_int"})
    assert checks(tmp_path, "task") == {"evaluator"}


def test_evaluator_with_rule_still_rejected(tmp_path: Path) -> None:
    make_task(tmp_path, "task", {"metric": "test_int", "rule": "larger_than"})
    assert checks(tmp_path, "task") == {"evaluator"}


def test_evaluator_test_bool_rejected(tmp_path: Path) -> None:
    make_task(tmp_path, "task", {"metric": "test_bool", "rule": "lambda x: x"})
    assert checks(tmp_path, "task") == {"evaluator"}


@pytest.mark.parametrize("prompt", [{"prompt": "", "data_source_type": "1"}, {"prompt": 7, "data_source_type": "1"}])
def test_prompt_value_rejected(tmp_path: Path, prompt: dict) -> None:
    make_task(tmp_path, "task", {}, prompt=prompt)
    assert checks(tmp_path, "task") == {"prompt"}


def test_prompt_is_not_object(tmp_path: Path) -> None:
    make_task(tmp_path, "task", {})
    (tmp_path / "data" / "task" / "prompt.json").write_text("[1, 2]")
    assert checks(tmp_path, "task") == {"prompt"}


def test_prompt_is_not_json(tmp_path: Path) -> None:
    make_task(tmp_path, "task", {})
    (tmp_path / "data" / "task" / "prompt.json").write_text("{oops")
    assert checks(tmp_path, "task") == {"prompt"}


def test_empty_tmc_list(tmp_path: Path) -> None:
    make_task(tmp_path, "task", {})
    (tmp_path / "metric" / "task" / "metric.yaml").write_text(yaml.dump({"TMC-list": []}))
    assert checks(tmp_path, "task") == {"metric-schema"}


def test_metric_yaml_is_not_object(tmp_path: Path) -> None:
    make_task(tmp_path, "task", {})
    (tmp_path / "metric" / "task" / "metric.yaml").write_text("- 1\n- 2\n")
    assert checks(tmp_path, "task") == {"metric-schema"}


def test_metric_yaml_is_not_parsable(tmp_path: Path) -> None:
    make_task(tmp_path, "task", {})
    (tmp_path / "metric" / "task" / "metric.yaml").write_text("TMC-list: [\n  - a: :\n")
    assert checks(tmp_path, "task") == {"metric-file"}


def test_tmc_item_is_not_dict(tmp_path: Path) -> None:
    make_task(tmp_path, "task", {})
    (tmp_path / "metric" / "task" / "metric.yaml").write_text(yaml.dump({"TMC-list": ["oops"]}))
    assert checks(tmp_path, "task") == {"metric-schema"}


def test_tmc_item_missing_keys(tmp_path: Path) -> None:
    make_task(tmp_path, "task", {})
    (tmp_path / "metric" / "task" / "metric.yaml").write_text(yaml.dump({"TMC-list": [{"code": GOOD_CODE}]}))
    assert checks(tmp_path, "task") == {"metric-schema"}
