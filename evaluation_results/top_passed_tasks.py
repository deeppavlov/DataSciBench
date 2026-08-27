import csv
from collections import defaultdict


def main():
    passed_models = defaultdict(set)
    all_tasks = set()

    with open("combined_results.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_tasks.add(row["data_name"])
            if row["task_name"] == "Completion Rate":
                try:
                    cr = float(row["result_cr"])
                    if cr == 1.0:
                        passed_models[row["data_name"]].add(row["model_name"])
                except ValueError:
                    pass

    results = [(task, len(passed_models[task])) for task in all_tasks]

    results.sort(key=lambda x: x[1], reverse=True)

    for task, count in results:
        print(f"{task}: {count}")


if __name__ == "__main__":
    main()
