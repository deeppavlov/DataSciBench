import os
import re

import pandas as pd

reports_dir = "/home/kaneki/Documents/DeepPavlov/подбор бенчмарка/DataSciBench/log_analysis/reports"
missing_dir = "/home/kaneki/Documents/DeepPavlov/подбор бенчмарка/DataSciBench/log_analysis/reports_missing"
csv_path = "/home/kaneki/Documents/DeepPavlov/подбор бенчмарка/DataSciBench/evaluation_results/our_models_orig.csv"

# 1. Извлечь имена задач из файлов
tasks: set[str] = set()
pattern = re.compile(r"(csv_excel_\d+|human_\d+|dl_\d+)")


def extract_tasks_from_dir(d):
    for f in os.listdir(d):
        match = pattern.search(f)
        if match:
            tasks.add(match.group(1))


extract_tasks_from_dir(reports_dir)
extract_tasks_from_dir(missing_dir)

print(f"Найдено уникальных задач в папках: {len(tasks)}")
# print(tasks)

# 2. Прочитать CSV
df = pd.read_csv(csv_path)

# Фильтруем строки по result_type == 'Completion Rate'
# Обрабатываем result_value как числа для безопасного сравнения (1 и 1.0)
df_cr = df[df["result_type"] == "Completion Rate"].copy()
df_cr["result_value_float"] = pd.to_numeric(df_cr["result_value"], errors="coerce")

# 3. Для каждой задачи из папок проверяем условие
results = []
for task in sorted(tasks):
    # Ищем строки для этой задачи (data_name), где result_value == 1
    task_runs = df_cr[(df_cr["data_name"] == task) & (df_cr["result_value_float"] == 1.0)]

    has_perfect_run = len(task_runs) > 0
    results.append({"Задача": task, "Есть успешный ран (CR=1)": "Да" if has_perfect_run else "Нет"})

res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))

# Сохраним также в CSV для удобства
res_df.to_csv("cr_check_results.csv", index=False)
print("Результаты сохранены в cr_check_results.csv")
