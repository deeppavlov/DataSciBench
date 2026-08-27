import argparse
import ast
import json
import logging
import shutil
from pathlib import Path

import yaml

from ..core.config import get_settings

logger = logging.getLogger(__name__)


def _parse_metrics_py(metrics_path: Path) -> list[dict]:
    content = metrics_path.read_text(encoding="utf-8")
    tree = ast.parse(content)

    metrics = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            for decor in node.decorator_list:
                if isinstance(decor, ast.Call) and getattr(decor.func, "id", "") == "metric":
                    entry = {}
                    for kw in decor.keywords:
                        if isinstance(kw.value, ast.Constant):
                            key = kw.arg
                            if key == "metric_name":
                                key = "metric"
                            entry[key] = kw.value.value
                    source = ast.get_source_segment(content, node)
                    if source:
                        entry["code"] = source
                    metrics.append(entry)

    if not metrics:
        raise ValueError(f"Cannot find any @metric decorators in {metrics_path}")

    return metrics


def _prompt_md_to_json(prompt_md_path: Path) -> dict:
    prompt_text = prompt_md_path.read_text(encoding="utf-8")
    return {
        "prompt": prompt_text,
        "data_source_type": "4=auto generated",
    }


def _metrics_to_yaml(metrics: list[dict], prompt_text: str) -> dict:
    tmc_list = []
    for m in metrics:
        entry = {
            "code": m["code"].strip(),
            "function": m["function"],
            "metric": m["metric"],
            "task_name": m["task_name"],
        }
        if m.get("ground_truth"):
            entry["ground_truth"] = m["ground_truth"]
        tmc_list.append(entry)

    return {
        "TMC-list": tmc_list,
        "prompt": prompt_text,
    }


def pack_single_task(task_dir: Path, task_id: str, benchmark_root: Path) -> None:
    prompt_md = task_dir / "prompt.md"
    metrics_py = task_dir / "metrics.py"
    gt_dir = task_dir / "gt"

    if not prompt_md.exists():
        raise FileNotFoundError(f"prompt.md not found in {task_dir}")
    if not metrics_py.exists():
        raise FileNotFoundError(f"metrics.py not found in {task_dir}")

    data_dir = benchmark_root / "data" / task_id
    metric_dir = benchmark_root / "metric" / task_id

    data_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    prompt_json = _prompt_md_to_json(prompt_md)
    (data_dir / "prompt.json").write_text(json.dumps(prompt_json, ensure_ascii=False), encoding="utf-8")
    logger.info("Created data/%s/prompt.json", task_id)

    if gt_dir.exists():
        dest_gt = data_dir / "gt"
        if dest_gt.exists():
            shutil.rmtree(dest_gt)
        shutil.copytree(gt_dir, dest_gt)
        logger.info("Copied gt/ -> data/%s/gt/", task_id)

    for f in task_dir.iterdir():
        if f.is_file() and f.suffix in (".csv", ".xlsx", ".npy", ".npz", ".json") and f.name != "prompt.json":
            shutil.copy2(f, data_dir / f.name)
            logger.info("Copied %s -> data/%s/%s", f.name, task_id, f.name)

    class BlockStringDumper(yaml.Dumper):
        pass

    def str_presenter(dumper: yaml.Dumper, data: str) -> yaml.ScalarNode:
        if "\n" in data:
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
        return dumper.represent_scalar("tag:yaml.org,2002:str", data)

    BlockStringDumper.add_representer(str, str_presenter)

    prompt_text = prompt_md.read_text(encoding="utf-8")
    metrics = _parse_metrics_py(metrics_py)
    metric_yaml = _metrics_to_yaml(metrics, prompt_text)
    yaml_path = metric_dir / "metric.yaml"
    yaml_path.write_text(
        yaml.dump(metric_yaml, Dumper=BlockStringDumper, allow_unicode=True, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    logger.info("Created metric/%s/metric.yaml", task_id)


def pack_all_tasks(output_dir: Path, prefix: str, benchmark_root: Path) -> None:
    task_dirs = sorted(d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("task_"))
    if not task_dirs:
        logger.warning("No task directories found in %s", output_dir)
        return

    for i, task_dir in enumerate(task_dirs):
        task_id = f"{prefix}_{i}"
        logger.info("Packing %s as %s", task_dir.name, task_id)
        pack_single_task(task_dir, task_id, benchmark_root)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Pack generated tasks into benchmark format")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task_dir", type=Path)
    group.add_argument("--output_dir", type=Path)
    parser.add_argument("--task_id", type=str)
    parser.add_argument("--prefix", type=str, default="custom")
    parser.add_argument("--benchmark_dir", type=Path, default=get_settings().benchmark_root)
    parser.add_argument("--mcp_tools", type=Path)
    args = parser.parse_args()

    if args.task_dir:
        if not args.task_id:
            parser.error("--task_id is required when using --task_dir")
        pack_single_task(args.task_dir, args.task_id, args.benchmark_dir)
        print(f"Packed {args.task_dir.name} as {args.task_id}")
    else:
        pack_all_tasks(args.output_dir, args.prefix, args.benchmark_dir)

    if args.mcp_tools and args.mcp_tools.exists():
        dest = args.benchmark_dir / "code_mode" / "_mcp_tools.py"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.mcp_tools, dest)
        logger.info("Copied _mcp_tools.py -> %s", dest)


if __name__ == "__main__":
    main()
