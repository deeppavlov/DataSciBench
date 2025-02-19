import asyncio
from dataclasses import asdict
# from metagpt.roles.di.data_interpreter import DataInterpreter
from role import SciDataInterpreter
from src.logs import create_logger, get_model_name
from metagpt.logs import logger
import os
from src.utils import change_dir, change_metalog_path
import argparse
import time
import json
from src.schemas import SciAgentBenchOutput
from metagpt.config2 import Config

SPECIFY_PATH_PROMPT = "All the input source data is at the `../` folder. And all the output files should be saved at the currect folder `./`.\n\n"

BCB_OUTPUT_PROMPT = "Complete the following instructions and write the final python code in a file named `output.py` using with open(...)\n\n"

def get_args():
    parser = argparse.ArgumentParser(description="Run the SciDataInterpreter role")
    parser.add_argument("--task_id", type=str, help="Specify the task id. If not specified, all tasks will be run.")
    parser.add_argument("--data_source_type", type=str, help="Specify the data source. If not specified, all tasks will be run.")
    parser.add_argument("--max_runs", type=int, default=3, help="Maximum running times")
    parser.add_argument("--gt_prompt", type=str, help="Specify the ground truth prompt")
    parser.add_argument("--continue_gen", action="store_true", help="Continue the previous run")
    parser.add_argument("--output_dir", type=str, help="Specify the output directory")
    parser.add_argument("--data_type", type=str, default="human", help="Specify the data type, including `csv`, `human`, `dl`, `bcb`")
    parser.add_argument("--skip_bcb", action="store_true", help="Skip the BCB data")
    # for scigentbench role
    parser.add_argument("--use_reflection", action="store_true", help="Use reflection")
    parser.add_argument("--hard_retry", action="store_true", help="Hard retry")
    parser.add_argument("--max_retry", type=int, default=3, help="Maximum retry times")
    parser.add_argument("--use_react", action="store_true", help="Use plan")
    # for customized config
    parser.add_argument("--config", default="test_config.yaml", type=str, help="Specify the config path")
    return parser.parse_args()

async def main(requirement: str, args=None):
    react_mode = "react" if args.use_react else "plan_and_act"
    config = Config.from_home(args.config)
    # config = Config.from_sab_config(args.config)
    role = SciDataInterpreter(
        use_reflection=args.use_reflection,
        hard_retry=args.hard_retry,
        max_retry=args.max_retry,
        react_mode=react_mode,
        config=config
    )
    print(role.llm)
    print(role.llm.config)
    # print(role.actions[0].llm.config)
    role.actions[0].llm.config = config.llm
    role.planner.set_plan_writter(config)
    # print(role)
    # print(role.config.llm)
    await role.run(requirement)

    return role.get_results_for_eval()

if __name__ == "__main__":
    # async def load_files():
    data_dir = "data/"
    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    num_folders = len(folders)
    args = get_args()
    if args.task_id is None:
        task_id = folders
    else:
        task_id = args.task_id if "[" not in args.task_id else eval(args.task_id)

    if isinstance(task_id, str):
        folders = [f"{args.task_id}"]
        num_folders = 1
    elif isinstance(task_id, list):
        folders = task_id
    else:
        pass

    # filter by data_source
    NaN = ''
    data_source_type = args.data_source_type
    filtered_folders = []
    for folder in folders:
        prompt_file = os.path.join(data_dir, folder, "prompt.json")
        if not os.path.exists(prompt_file):
            continue
        with open(prompt_file, "r") as file:
            prompt_data = eval(file.read())
            if data_source_type is None or prompt_data["data_source_type"].startswith(data_source_type):
                filtered_folders.append(folder)
    # folders=['bcb1011', 'bcb1017', 'bcb102', 'bcb1024', 'bcb1035', 'bcb1037', 'bcb104', 'bcb1043', 'bcb105', 'bcb1051']

    
    folders=filtered_folders
    # folders = list(reversed(filtered_folders))
    # print(folders)
    for id, folder in enumerate(folders):
        prompt_file = os.path.join(data_dir, folder, "prompt.json")
        # separately save each run
        for sub_idx in range(args.max_runs):
            with open(prompt_file, "r") as file:
                prompt_data = eval(file.read())
            # ===================================================
            # if orig_log_dir has already finished, skip
            # _, _, orig_log_dir, _ = create_logger(folder)
            # if os.path.getsize(os.path.join(orig_log_dir, "logs.txt")) != 0:
            #     print("Skipping folder", folder)
            #     continue
            # ===================================================
            if folder.startswith('bcb'):
                result_logger, time_logger, log_dir, run_dir = create_logger(folder, sub_idx, config_name=args.config, split=False)
            else:
                result_logger, time_logger, log_dir, run_dir = create_logger(folder, sub_idx, config_name=args.config, split=True)
            orig_log_file_path = os.path.join(log_dir)
            log_file_path = os.path.join(log_dir, "logs.txt")
            sys_log_file_path = os.path.join(log_dir, "sys_logs.txt")
            # sys_log_file_path = os.path.join(log_dir, "sys_logs.txt")

            # if os.path.getsize(sys_log_file_path) != 0:
            sys_log = ""
            if os.path.exists(sys_log_file_path):
                with open(sys_log_file_path, "r") as f:
                    sys_log = str(f.read())
            if "JSONDecodeError" in sys_log and "chatanywhere_error" not in sys_log:
                print("Skipping folder", folder)
                continue
            if os.path.getsize(log_file_path) != 0 and not args.continue_gen:
                print("Skipping folder", folder)
                continue

            # save misc. model statistics
            model_name = get_model_name(args.config)
            if not folder.startswith('bcb'):
                model_name = model_name.split("/")[-1]
            output_dict_path = os.path.join(run_dir, f"{model_name}_outputs.jsonl")
            output_dict_path = os.path.abspath(output_dict_path)
            # specify where to load data and save data
            if not prompt_data["data_source_type"].startswith("1"):
                requirement = SPECIFY_PATH_PROMPT + prompt_data["prompt"]
            else:
                requirement = prompt_data["prompt"]
            # if 'bcb' in folder:
            #     requirement = BCB_OUTPUT_PROMPT + requirement
            if 'bcb' in folder and args.skip_bcb:
                print(f"Skipping {folder}")
                continue
            if args.data_type not in folder:
                print(f"Skipping {folder}")
                continue
            if args.gt_prompt is not None:
                requirement = args.gt_prompt + '\n' + requirement
            # with change_dir(log_dir):
            sys_output_path = os.path.join(log_dir, "sys_logs.txt")
            # sys_output_path = os.path.abspath(sys_output_path)  # to avoid conflict with the change_dir context manager
            print(sys_output_path)
            print(output_dict_path)
            # redirect the log path for the metagpt logger
            with change_metalog_path(logger=logger, file_path=sys_output_path) as temp_logger:
                # redirect the pwd to the data folder
                with change_dir(log_dir):
                    try:
                        temp_logger.info(f"Processing {folder} ({id}/{num_folders})")
                        temp_logger.info(f"Prompt:\n{requirement}")
                        
                        time_logger.info(f"Processing {folder} ({id}/{num_folders})")
                        start_time = time.time()
                        plan_list, cost_list, error_counter_list = asyncio.run(main(requirement, args))
                        end_time = time.time()
                        elapsed_time = end_time - start_time

                        temp_logger.info(f"Completed processing folder {folder} ({id+1}/{num_folders})")
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
                            plan=plan_list[-1]
                        )
                        output_dict = asdict(output_dict)

                        with open(output_dict_path, "a") as f:
                            f.write(json.dumps(output_dict)+'\n')

                    except Exception as e:
                        temp_logger.info("====================================================")
                        temp_logger.info(f"{e}\n====================================================")
