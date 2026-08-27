import asyncio
import os
from dataclasses import asdict
from typing import Any, Literal

from metagpt.logs import logger

os.environ["MPLBACKEND"] = "Agg"

import argparse
import json
import time

from metagpt.config2 import Config
from src.logs import create_logger, get_model_name
from src.schemas import SciAgentBenchOutput
from src.utils import change_dir, change_metalog_path

SPECIFY_PATH_PROMPT = "All the input source data is at the `../` folder. And all the output files should be saved at the currect folder `./`.\n\n"


async def main_simple(requirement: str, config: Config):
    from experiments.simple_runner import run_task

    result = await run_task(requirement, config)

    plan_list: list[list[Any]] = [[]]
    cost_list = [[0, 0, 0, 0]]
    error_counter_list = [[result.error_count]]

    return plan_list, cost_list, error_counter_list


async def main_legacy(requirement: str, args):
    from role import SciDataInterpreter

    react_mode: Literal["plan_and_act", "react"] = "react" if args.use_react else "plan_and_act"
    config = Config.from_home(args.config)
    role = SciDataInterpreter(
        use_reflection=args.use_reflection,
        hard_retry=args.hard_retry,
        max_retry=args.max_retry,
        react_mode=react_mode,
        config=config,  # type: ignore[call-arg]
    )
    role.actions[0].llm.config = config.llm
    role.planner.set_plan_writter(config)
    await role.run(requirement)
    return role.get_results_for_eval()


def get_args():
    parser = argparse.ArgumentParser(description="Run benchmark tasks")
    parser.add_argument("--task_id", type=str)
    parser.add_argument("--data_source_type", type=str)
    parser.add_argument("--max_runs", type=int, default=3)
    parser.add_argument("--gt_prompt", type=str)
    parser.add_argument("--continue_gen", action="store_true")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--data_type", type=str, default="human")
    parser.add_argument("--skip_bcb", action="store_true")
    parser.add_argument("--use_reflection", action="store_true")
    parser.add_argument("--hard_retry", action="store_true")
    parser.add_argument("--max_retry", type=int, default=3)
    parser.add_argument("--use_react", action="store_true")
    parser.add_argument("--config", default="test_config.yaml", type=str)
    parser.add_argument("--legacy", action="store_true", help="Use legacy MetaGPT SciDataInterpreter mode")
    return parser.parse_args()


if __name__ == "__main__":
    data_dir = "data/"
    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    num_folders = len(folders)
    args = get_args()
    task_id = folders if args.task_id is None else args.task_id if "[" not in args.task_id else eval(args.task_id)

    if isinstance(task_id, str):
        folders = [f"{args.task_id}"]
        num_folders = 1
    elif isinstance(task_id, list):
        folders = task_id

    NaN = ""
    data_source_type = args.data_source_type
    filtered_folders = []
    for folder in folders:
        prompt_file = os.path.join(data_dir, folder, "prompt.json")
        if not os.path.exists(prompt_file):
            continue
        with open(prompt_file) as file:
            prompt_data = eval(file.read())
            ds_type = prompt_data.get("data_source_type", "None")
            if data_source_type is None or str(ds_type).startswith(str(data_source_type)):
                filtered_folders.append(folder)

    folders = filtered_folders

    for folder_idx, folder in enumerate(folders):
        if "bcb" in folder and args.skip_bcb:
            continue
        if args.data_type not in folder:
            continue

        prompt_file = os.path.join(data_dir, folder, "prompt.json")

        for sub_idx in range(args.max_runs):
            with open(prompt_file) as file:
                prompt_data = eval(file.read())

            if folder.startswith("bcb"):
                result_logger, time_logger, log_dir, run_dir = create_logger(
                    folder, sub_idx, config_name=args.config, split=False
                )
            else:
                result_logger, time_logger, log_dir, run_dir = create_logger(
                    folder, sub_idx, config_name=args.config, split=True
                )

            log_file_path = os.path.join(log_dir, "logs.txt")
            sys_log_file_path = os.path.join(log_dir, "sys_logs.txt")

            sys_log = ""
            if os.path.exists(sys_log_file_path):
                with open(sys_log_file_path) as f:
                    sys_log = str(f.read())
            if "JSONDecodeError" in sys_log and "chatanywhere_error" not in sys_log:
                print("Skipping folder", folder)
                continue
            if os.path.getsize(log_file_path) != 0 and not args.continue_gen:
                print("Skipping folder", folder)
                continue

            model_name = get_model_name(args.config)
            if not folder.startswith("bcb"):
                model_name = model_name.split("/")[-1]
            output_dict_path = os.path.join(run_dir, f"{model_name}_outputs.jsonl")
            output_dict_path = os.path.abspath(output_dict_path)

            requirement = SPECIFY_PATH_PROMPT + prompt_data["prompt"]
            if args.gt_prompt is not None:
                requirement = args.gt_prompt + "\n" + requirement

            sys_output_path = os.path.join(log_dir, "sys_logs.txt")
            print(sys_output_path)
            print(output_dict_path)

            with change_metalog_path(logger=logger, file_path=sys_output_path) as temp_logger, change_dir(log_dir):
                try:
                    temp_logger.info(f"Processing {folder} ({folder_idx}/{num_folders})")
                    temp_logger.info(f"Prompt:\n{requirement}")

                    temp_logger.info("=== ENVIRONMENT DIAGNOSTICS ===")
                    temp_logger.info(f"CWD: {os.getcwd()}")
                    temp_logger.info(f"Files in CWD: {os.listdir('.')}")
                    try:
                        temp_logger.info(f"Files in parent dir: {os.listdir('..')}")
                    except Exception as e:
                        temp_logger.info(f"Cannot list parent dir: {e}")
                    temp_logger.info(f"data_source_type: {prompt_data.get('data_source_type', 'N/A')}")
                    temp_logger.info(f"Mode: {'legacy (MetaGPT)' if args.legacy else 'simple (single snippet)'}")
                    temp_logger.info("=== END DIAGNOSTICS ===")

                    time_logger.info(f"Processing {folder} ({folder_idx}/{num_folders})")
                    start_time = time.time()

                    if args.legacy:
                        plan_list, cost_list, error_counter_list = asyncio.run(main_legacy(requirement, args))
                    else:
                        config = Config.from_home(args.config)
                        plan_list, cost_list, error_counter_list = asyncio.run(main_simple(requirement, config))

                    end_time = time.time()
                    elapsed_time = end_time - start_time

                    temp_logger.info(f"Completed processing folder {folder} ({folder_idx + 1}/{num_folders})")
                    temp_logger.info(f"Plan list:\n{plan_list}")
                    temp_logger.info(f"Cost list:\n{cost_list}")
                    temp_logger.info(f"Error counter list:\n{error_counter_list}")

                    result_logger.info(f"Plan list:\n{plan_list}")
                    result_logger.info(f"Cost list:\n{cost_list}")
                    result_logger.info(f"Error counter list:\n{error_counter_list}")

                    time_logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")

                    output_dict = SciAgentBenchOutput(
                        output_dir=log_dir,
                        time_cost=elapsed_time,
                        error_list=error_counter_list[-1],
                        cost=cost_list[-1],
                        plan=plan_list[-1],
                    )
                    output_json = asdict(output_dict)

                    with open(output_dict_path, "a") as f:
                        f.write(json.dumps(output_json) + "\n")

                except Exception:
                    import traceback

                    temp_logger.info("====================================================")
                    temp_logger.info(f"{traceback.format_exc()}\n====================================================")
