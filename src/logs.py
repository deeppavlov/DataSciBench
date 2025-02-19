import logging
import argparse
import os
import yaml
from metagpt.const import CONFIG_ROOT

def get_model_name(config_name="config2.yaml"):
    assert config_name != "config2.yaml", "Please specify a config file name instead of using the default one for now"
    print("Config Root: ", CONFIG_ROOT)
    print("Config Name: ", config_name)
    config_path = os.path.join(CONFIG_ROOT, config_name)
    # Load the config file
    config_file = os.path.expanduser(config_path)
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Get the model name from the config
    model_name = config["llm"]["model"]

    return model_name

def create_logger(id, sub_idx=None, target_dir=None, config_name=None, split=True):
    # Create the log directory if it doesn't exist
    if split:
        model_name = get_model_name(config_name).split('/')[-1]
    else:
        model_name = get_model_name(config_name)
    if sub_idx is not None:
        log_dir = os.path.join("data", str(id), model_name+f"_{sub_idx}")
        run_dir = os.path.join("data", str(id))
        os.makedirs(log_dir, exist_ok=True)
        time_log_file = os.path.join("logs", model_name+f"_{sub_idx}_time.txt")
    else:
        log_dir = os.path.join("data", str(id), model_name)
        run_dir = os.path.join("data", str(id))
        os.makedirs(log_dir, exist_ok=True)
        time_log_file = os.path.join("logs", model_name+"_time.txt")

    if target_dir is not None:
        log_dir = os.path.join("data", str(id), target_dir)
        run_dir = os.path.join("data", str(id))
        os.makedirs(log_dir, exist_ok=True)
        time_log_file = os.path.join("logs", target_dir+"_time.txt")

    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create a file handler and set the log file path
    log_file = os.path.join(log_dir, "logs.txt")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter and add it to the file handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    if logger.hasHandlers():
        logger.handlers.clear()
    # Add the file handler to the logger
    logger.addHandler(file_handler)

    # ===================================================
    # Create the time logger below
    # ===================================================

    time_logger = logging.getLogger(f"time_logger")
    time_logger.setLevel(logging.INFO)

    # Create a file handler for the time log
    time_file_handler = logging.FileHandler(time_log_file)
    time_file_handler.setLevel(logging.INFO)

    # Use a simple formatter for the time log
    time_formatter = logging.Formatter('%(asctime)s - %(message)s')
    time_file_handler.setFormatter(time_formatter)

    # Clear existing handlers and add the new time file handler
    if time_logger.hasHandlers():
        time_logger.handlers.clear()
    time_logger.addHandler(time_file_handler)

    return logger, time_logger, log_dir, run_dir

if __name__ == "__main__":
    logger = create_logger(0)
    logger.info("Logger created")