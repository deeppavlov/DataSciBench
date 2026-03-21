import pandas as pd
import os
from collections import defaultdict

data_orig_path = '/home/kaneki/Documents/DeepPavlov/подбор бенчмарка/DataSciBench/data'
csv_path = '/home/kaneki/Documents/DeepPavlov/подбор бенчмарка/DataSciBench/evaluation_results/our_models_orig.csv'

models = {
    'gemma-3-27b-it': 10,
    'Qwen3.5-27B': 3,
    'Qwen3-30B-A3B': 3
}

def main():
    all_tasks = sorted([d for d in os.listdir(data_orig_path) if os.path.isdir(os.path.join(data_orig_path, d))])
    
    df = pd.read_csv(csv_path)
    
    df = df[df['model_name'] != 'model_name'].copy()
    
    df['run_id'] = pd.to_numeric(df['run_id'], errors='coerce')
    df = df.dropna(subset=['run_id'])
    df['run_id'] = df['run_id'].astype(int)
    
    missing = []

    for model, expected_runs in models.items():
        model_df = df[df['model_name'] == model]
        for task in all_tasks:
            task_df = model_df[model_df['data_name'] == task]
            existing_runs = task_df['run_id'].unique()
            for run_id in range(expected_runs):
                if run_id not in existing_runs:
                    missing.append({
                        'model': model,
                        'task': task,
                        'run_id': run_id
                    })

    if not missing:
        print("Все задачи и раны на месте.")
        return

    print(f"Найдено {len(missing)} отсутствующих ранов.")
    
    grouped = defaultdict(lambda: defaultdict(list))
    for m in missing:
        grouped[m['model']][m['task']].append(m['run_id'])
    
    for model in sorted(grouped.keys()):
        print(f"\nМодель: {model}")
        for task in sorted(grouped[model].keys()):
            runs = grouped[model][task]
            print(f"  - {task}: {runs}")

if __name__ == "__main__":
    main()
