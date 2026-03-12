import os
import json
import matplotlib.pyplot as plt

data_dir = '/home/kaneki/Documents/DeepPavlov/подбор бенчмарка/DataSciBench/data1'
results = []

for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file == 'Qwen3-30B-A3B_outputs.jsonl':
            filepath = os.path.join(root, file)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        data = json.loads(line)
                        if 'time_cost' in data:
                            out_dir = data.get('output_dir', '')
                            # Extract dl_10/gemma-3-27b-it_1 from data/dl_10/gemma-3-27b-it_1
                            parts = out_dir.replace('\\', '/').split('/')
                            if len(parts) >= 2:
                                task_display = f"{parts[-2]}/{parts[-1]}"
                            else:
                                task_display = os.path.basename(root)
                            
                            results.append({'task': task_display, 'time_cost': data['time_cost']})
            except Exception as e:
                print(f"Error reading {filepath}: {e}")

# Sort descending
results.sort(key=lambda x: x['time_cost'], reverse=True)

long_runs = [r for r in results if r['time_cost'] > 1000]
print(f"All runs longer than 1000 seconds ({len(long_runs)} items):")
for i, res in enumerate(long_runs):
    print(f"{i+1}. Task: {res['task']}, Time: {res['time_cost']:.2f} sec")

# Draw scatter plot
times = [r['time_cost'] for r in results]
x_vals = list(range(len(times)))

plt.figure(figsize=(12, 6))
plt.scatter(x_vals, times, color='blue', alpha=0.6, s=50)
plt.title('Execution Time Scatter Plot (time_cost)')
plt.ylabel('Time (sec)')
plt.xlabel('Tasks (sorted by time descending)')

# Labels for outliers (> 1000 sec)
for i, res in enumerate(results):
    if res['time_cost'] > 1000:
        plt.annotate(res['task'], (x_vals[i], times[i]), 
                     xytext=(5, 5), textcoords='offset points', 
                     fontsize=9, color='red')

output_img = '/home/kaneki/Documents/DeepPavlov/подбор бенчмарка/DataSciBench/time_costs_scatter.png'
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(output_img)
plt.close()

print(f"\nPlot saved to: {output_img}")
