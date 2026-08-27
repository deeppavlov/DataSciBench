import re
from collections import Counter


def analyze_log(file_path):
    print(f"--- Analysis for {file_path} ---")
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    # Count occurrences of errors
    errors = re.findall(r"ERROR.*?\| (.*)", content)
    error_counter = Counter(errors)
    print("Top Errors:")
    for err, count in error_counter.most_common(5):
        print(f"  {count}: {err[:100]}")

    # Count task completions or processing steps
    tasks = re.findall(r"Processing (human_\d+)", content)
    print(f"Total tasks processed: {len(tasks)}")

    failed_tasks = re.findall(r"Task failed: (human_\d+)", content)
    print(f"Failed tasks explicit: {len(failed_tasks)}")

    # Check for context limits or generation cutoffs
    cutoffs = len(re.findall(r"finish_reason.*?(?:length|max_tokens|cut off)", content.lower()))
    print(f"Possible generation cutoffs (finish_reason=length): {cutoffs}")

    # Check for execution errors (MetaGPT often prints these from python execution)
    syntax_errors = len(re.findall(r"SyntaxError", content))
    indent_errors = len(re.findall(r"IndentationError", content))
    name_errors = len(re.findall(r"NameError", content))
    attribute_errors = len(re.findall(r"AttributeError", content))

    print(f"SyntaxErrors: {syntax_errors}")
    print(f"IndentationErrors: {indent_errors}")
    print(f"NameErrors: {name_errors}")
    print(f"AttributeErrors: {attribute_errors}")
    print()


analyze_log("/home/kaneki/Documents/DeepPavlov/подбор бенчмарка/DataSciBench/log_archive/Qwen3.5-27B.log")
analyze_log("/home/kaneki/Documents/DeepPavlov/подбор бенчмарка/DataSciBench/log_archive/context/Qwen3.5-27B.log")
