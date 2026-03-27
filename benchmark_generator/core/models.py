from pydantic import BaseModel, Field


class GeneratedTask(BaseModel):
    prompt: str
    needs_input_data: bool = False
    input_data_code: str | None = None


class TaskList(BaseModel):
    tasks: list[GeneratedTask]


class SubTask(BaseModel):
    name: str
    description: str
    output_files: list[str] = Field(default_factory=list)


class Solution(BaseModel):
    code: str
    subtasks: list[SubTask]


class MetricEntry(BaseModel):
    task_name: str
    function: str
    metric: str
    ground_truth: str | None = None
    code: str


class MetricsList(BaseModel):
    metrics: list[MetricEntry]
