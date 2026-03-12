import json
from pathlib import Path

from benchmark_generator.models import GeneratedTask, TaskList


class TestPromptMdGeneration:
    def test_prompt_written_correctly(self, tmp_task_dir, sample_prompt_md):
        prompt_path = tmp_task_dir / "prompt.md"
        prompt_path.write_text(sample_prompt_md)
        content = prompt_path.read_text()
        assert "Train a classifier" in content
        assert "data_analysis.txt" in content

    def test_input_data_py_written(self, tmp_task_dir):
        code = "import pandas as pd\npd.DataFrame({'a': [1,2]}).to_csv('data.csv', index=False)"
        task = GeneratedTask(prompt="test", needs_input_data=True, input_data_code=code)

        if task.needs_input_data and task.input_data_code:
            (tmp_task_dir / "input_data.py").write_text(task.input_data_code)

        assert (tmp_task_dir / "input_data.py").exists()
        assert "pandas" in (tmp_task_dir / "input_data.py").read_text()

    def test_no_input_data_py_when_not_needed(self, tmp_task_dir):
        task = GeneratedTask(prompt="test")
        if task.needs_input_data and task.input_data_code:
            (tmp_task_dir / "input_data.py").write_text(task.input_data_code)
        assert not (tmp_task_dir / "input_data.py").exists()


class TestTaskListParsing:
    def test_roundtrip_json(self):
        original = TaskList(
            tasks=[
                GeneratedTask(prompt="Task 1"),
                GeneratedTask(prompt="Task 2", needs_input_data=True, input_data_code="pass"),
            ]
        )
        json_str = original.model_dump_json()
        restored = TaskList.model_validate_json(json_str)
        assert len(restored.tasks) == 2
        assert restored.tasks[0].prompt == "Task 1"
        assert restored.tasks[1].input_data_code == "pass"
