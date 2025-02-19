#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modified DI role for DataSciBench
@Modified by: 2024/8/6. Added a plan_list to capture the completed/failed plans.
"""
from __future__ import annotations

import json
from typing import Literal

from pydantic import Field, model_validator

from metagpt.actions.di.ask_review import ReviewConst
from metagpt.actions.di.execute_nb_code import ExecuteNbCode
from metagpt.actions.di.write_analysis_code import CheckData, WriteAnalysisCode
from metagpt.logs import logger
from metagpt.prompts.di.write_analysis_code import DATA_INFO
from metagpt.roles import Role
from metagpt.schema import Message, Task, TaskResult, Plan
from metagpt.strategy.task_type import TaskType
from metagpt.tools.tool_recommend import BM25ToolRecommender, ToolRecommender
from metagpt.utils.common import CodeParser

from metagpt.utils.cost_manager import Costs
from metagpt.config2 import Config

REACT_THINK_PROMPT = """
# User Requirement
{user_requirement}
# Context
{context}

Output a json following the format:
```json
{{
    "thoughts": str = "Thoughts on current situation, reflect on how you should proceed to fulfill the user requirement",
    "state": bool = "Decide whether you need to take more actions to complete the user requirement. Return true if you think so. Return false if you think the requirement has been completely fulfilled."
}}
```
"""


class SciDataInterpreter(Role):
    name: str = "SciAgent"
    profile: str = "DataInterpreter"
    auto_run: bool = True
    use_plan: bool = True
    use_reflection: bool = False
    execute_code: ExecuteNbCode = Field(default_factory=ExecuteNbCode, exclude=True)
    tools: list[str] = []  # Use special symbol ["<all>"] to indicate use of all registered tools
    tool_recommender: ToolRecommender = None
    react_mode: Literal["plan_and_act", "react"] = "plan_and_act"
    max_react_loop: int = 10  # used for react mode

    # For evaluation
    plan_list: list[Plan] = [] # a list of completed/failed plans, we can get the code and resutls from it
    cost_list: list[Costs] = [] # a list of costs for each (TODO:? not sure, might be the accumulated cost) plan
    step_error_counter_list: list[int] = [] # a temporary error counter for each plan
    error_counter_list: list[int] = [] # a list of error counter for each plan
    hard_retry: bool = False
    max_retry: int = 3
    # config: Config

    # def __init__(self, config):
    #     super().__init__(config)

    def update_results_for_eval(self, rsp: Plan):
        rsp_json = self.read_json_from_list(rsp)
        self.plan_list.append(rsp_json)
        self.cost_list.append(self.llm.cost_manager.get_costs())
        self.error_counter_list.append(self.step_error_counter_list)
        self.step_error_counter_list = []

    def update_react_results_for_eval(self, rsp: TaskResult):
        self.error_counter_list.append(self.step_error_counter_list)

    def get_results_for_eval(self):
        return self.plan_list, self.cost_list, self.error_counter_list

    # helper func 1
    def read_json_from_list(self, plan):
        content = str(plan)
        # Remove any non-JSON content
        json_start_pos = content.find("## Current Plan")
        json_end_pos = content.find("## Current Task")
        content = content[json_start_pos+16:json_end_pos]
        json_objects = json.loads(content)
        return json_objects

    @model_validator(mode="after")
    def set_plan_and_tool(self) -> "Interpreter":
        self._set_react_mode(react_mode=self.react_mode, max_react_loop=self.max_react_loop, auto_run=self.auto_run)
        self.use_plan = (
            self.react_mode == "plan_and_act"
        )  # create a flag for convenience, overwrite any passed-in value
        if self.tools and not self.tool_recommender:
            self.tool_recommender = BM25ToolRecommender(tools=self.tools)
        self.set_actions([WriteAnalysisCode])
        self._set_state(0)
        return self

    @property
    def working_memory(self):
        return self.rc.working_memory

    async def _think(self) -> bool:
        """Useful in 'react' mode. Use LLM to decide whether and what to do next."""
        user_requirement = self.get_memories()[0].content
        context = self.working_memory.get()

        if not context:
            # just started the run, we need action certainly
            self.working_memory.add(self.get_memories()[0])  # add user requirement to working memory
            self._set_state(0)
            return True

        prompt = REACT_THINK_PROMPT.format(user_requirement=user_requirement, context=context)
        rsp = await self.llm.aask(prompt)
        rsp_dict = json.loads(CodeParser.parse_code(block=None, text=rsp))
        self.working_memory.add(Message(content=rsp_dict["thoughts"], role="assistant"))
        need_action = rsp_dict["state"]
        self._set_state(0) if need_action else self._set_state(-1)

        return need_action

    async def _act(self) -> Message:
        """Useful in 'react' mode. Return a Message conforming to Role._act interface."""
        code, _, _ = await self._write_and_exec_code()
        return Message(content=code, role="assistant", cause_by=WriteAnalysisCode)

    async def _plan_and_act(self) -> Message:
        try:
            rsp = await super()._plan_and_act()
            self.update_results_for_eval(rsp)
            await self.execute_code.terminate()
            return rsp
        except Exception as e:
            await self.execute_code.terminate()
            raise e

    async def _act_on_task(self, current_task: Task) -> TaskResult:
        """Useful in 'plan_and_act' mode. Wrap the output in a TaskResult for review and confirmation."""
        code, result, is_success = await self._write_and_exec_code(max_retry=self.max_retry)
        task_result = TaskResult(code=code, result=result, is_success=is_success)
        self.update_react_results_for_eval(task_result)
        return task_result

    async def _write_and_exec_code(self, max_retry: int = 3):
        counter = 0
        error_counter = -1
        success = False

        # plan info
        plan_status = self.planner.get_plan_status() if self.use_plan else ""

        # tool info
        if self.tool_recommender:
            context = (
                self.working_memory.get()[-1].content if self.working_memory.get() else ""
            )  # thoughts from _think stage in 'react' mode
            plan = self.planner.plan if self.use_plan else None
            tool_info = await self.tool_recommender.get_recommended_tool_info(context=context, plan=plan)
        else:
            tool_info = ""

        # data info
        await self._check_data()

        while not success and counter < max_retry:
            ### write code ###
            code, cause_by = await self._write_code(counter, plan_status, tool_info)

            self.working_memory.add(Message(content=code, role="assistant", cause_by=cause_by))

            ### execute code ###
            result, success = await self.execute_code.run(code)

            self.working_memory.add(Message(content=result, role="user", cause_by=ExecuteNbCode))

            ### process execution result ###
            counter += 1
            error_counter += 1

            if not success and counter >= max_retry and not self.hard_retry:
                logger.info("coding failed!")
                review, _ = await self.planner.ask_review(auto_run=False, trigger=ReviewConst.CODE_REVIEW_TRIGGER)
                if ReviewConst.CHANGE_WORDS[0] in review:
                    counter = 0  # redo the task again with help of human suggestions

        self.step_error_counter_list.append(error_counter)
        return code, result, success

    async def _write_code(
        self,
        counter: int,
        plan_status: str = "",
        tool_info: str = "",
    ):
        todo = self.rc.todo  # todo is WriteAnalysisCode
        logger.info(f"ready to {todo.name}")
        use_reflection = counter > 0 and self.use_reflection  # only use reflection after the first trial

        user_requirement = self.get_memories()[0].content
        print("!!!Requirement:\n\n", user_requirement)

        code = await todo.run(
            user_requirement=user_requirement,
            plan_status=plan_status,
            tool_info=tool_info,
            working_memory=self.working_memory.get(),
            use_reflection=use_reflection,
        )

        return code, todo

    async def _check_data(self):
        if (
            not self.use_plan
            or not self.planner.plan.get_finished_tasks()
            or self.planner.plan.current_task.task_type
            not in [
                TaskType.DATA_PREPROCESS_CLEANING.type_name,
                TaskType.PREDICTIVE_MODELING.type_name,
                # TaskType.DATA_MINING.type_name,
                TaskType.PATTERN_RECOGNITION.type_name,
            ]
        ):
            return
        logger.info("Check updated data")
        code = await CheckData().run(self.planner.plan)
        if not code.strip():
            return
        result, success = await self.execute_code.run(code)
        if success:
            data_info = DATA_INFO.format(info=result)
            self.working_memory.add(Message(content=data_info, role="user", cause_by=CheckData))
