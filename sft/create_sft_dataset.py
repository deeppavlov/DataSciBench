import argparse
import ast
import json
import logging
import random
import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmark_generator.core.config import get_settings
from benchmark_generator.tools.generate_tasks import generate_tasks
from benchmark_generator.tools.solve_task import solve_tasks
from benchmark_generator.tools.pack_task import pack_single_task

logger = logging.getLogger(__name__)

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Generate SFT dataset 80/20 from a single codebase and topic")
    parser.add_argument("--codebase", type=Path, required=True, help="Path to codebase / documentation")
    parser.add_argument("--topic", type=Path, default=None, help="Path to topic file")
    parser.add_argument("--count", type=int, default=10, help="Number of total tasks to try generating")
    parser.add_argument("--output_dir", type=Path, default=Path("output/sft_gen"), help="Dir to store raw tasks temporarily")
    parser.add_argument("--sft_output_file", type=Path, default=Path("sft/sft_dataset.jsonl"), help="Path to output jsonl dataset")
    parser.add_argument("--test_prefix", type=str, default="sft_test", help="Prefix for benchmark task_ids")
    
    args = parser.parse_args()

    logger.info("=== STEP 1: Generating tasks ===")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated_dirs = generate_tasks(
        codebase_path=args.codebase,
        count=args.count,
        output_dir=args.output_dir,
        topic_path=args.topic,
        code_mode=False
    )
    
    logger.info("=== STEP 2: Solving and checking metrics ===")
    codebase_text = args.codebase.read_text(encoding="utf-8") if args.codebase.exists() else ""
    topic_text = args.topic.read_text(encoding="utf-8") if args.topic and args.topic.exists() else None
    
    solve_tasks(
        output_dir=args.output_dir,
        codebase_text=codebase_text,
        topic_text=topic_text,
        code_mode=False
    )

    logger.info("=== STEP 3: Filtering successful tasks ===")
    task_dirs = sorted(d for d in args.output_dir.iterdir() if d.is_dir() and d.name.startswith("task_"))
    successful_tasks = []
    
    for td in task_dirs:
        if (td / "verify_log.txt").exists():
            successful_tasks.append(td)
        else:
            logger.warning("Task %s failed metrics verification. Skipping from dataset.", td.name)

    logger.info(f"Successful tasks: {len(successful_tasks)} out of {len(task_dirs)}")

    if not successful_tasks:
        logger.error("No tasks passed testing. Cannot create dataset.")
        return

    if len(successful_tasks) < 2:
        logger.error("Too few tasks passed testing to make a split (need >=2).")
        raise ValueError("Слишком мало задач для корректного разделения на Train/Test.")
        
    random.seed(42)
    random.shuffle(successful_tasks)
    
    split_index = max(1, int(len(successful_tasks) * 0.8))
    if split_index == len(successful_tasks):
        split_index = len(successful_tasks) - 1
        
    train_tasks = successful_tasks[:split_index]
    test_tasks = successful_tasks[split_index:]
        
    logger.info(f"Split results: {len(train_tasks)} for TRAIN, {len(test_tasks)} for TEST (Benchmark)")

    logger.info("=== STEP 4: Packing TEST tasks to benchmark ===")
    settings = get_settings()
    for i, t_dir in enumerate(test_tasks):
        task_id = f"{args.test_prefix}_{i:03d}"
        logger.info(f"Packing {t_dir.name} as {task_id} to {settings.benchmark_root}")
        pack_single_task(t_dir, task_id, settings.benchmark_root)

    logger.info(f"=== STEP 5: Exporting TRAIN tasks to {args.sft_output_file} ===")
    args.sft_output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.sft_output_file, "w", encoding="utf-8") as f:
        for t_dir in train_tasks:
            prompt_md = t_dir / "prompt.md"
            solution_py = t_dir / "solution.py"
            
            if prompt_md.exists() and solution_py.exists():
                prompt_content = prompt_md.read_text(encoding="utf-8").strip()
                code_content = solution_py.read_text(encoding="utf-8").strip()
                
                user_content = (
                    "You are a professional Data Scientist. Write clean, working Python code to solve the given tasks. "
                    "Return ONLY the code.\n\n"
                    f"Task:\n{prompt_content}"
                )
                
                row = {
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": code_content}
                    ]
                }
                
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                
    logger.info("Done! SFT pipeline data generation completed.")

if __name__ == "__main__":
    main()
