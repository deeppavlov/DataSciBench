# DataSciBench: An LLM Agent Benchmark for Data Science

<p align="center">
📃 <a href="https://arxiv.org/abs/2502.13897" target="_blank">[DataSciBench]</a> 
<a href="https://github.com/THUDM/DataSciBench" target="_blank">[GitHub]</a>
<a href="https://huggingface.co/datasets/zd21/DataSciBench/tree/main" target="_blank">[Evaluation Data]</a>
<a href="https://datascibench.github.io/" target="_blank">[Website]</a> <br> <br>
</p>

## Introduction

### Install MetaGPT

```bash
cd MetaGPT
pip install .
```

### Data

All collected prompts have been processed through `notebooks/preprocess_prompt.ipynb` and save into `data/{task_id}/`.
   

### Prepare Env

Run
```bash
pip install -r requirements.txt
```
To run dl experiments, you may also need to log in to wandb.

### Experimental Setting

Config model in `~/.metagpt/config2.yaml/`

### Running

Run all experiments as follows
```bash
python -c "import sys; sys.path.append('/path/to/DataSciBench'); from experiments import run_examples"

python -m experiments.run_examples
```
Run a particular experiment, as follows
```bash
python -m experiments.run_examples --task_id dl_0

python -m experiments.run_examples --data_type dl --config test_config_dl.yaml
```
Specifies to run a prompt of some kind, for example, to run a prompt with no external data dependencies
```bash
python -m experiments.run_examples --data_source_type 1
```

### Check the generation process
```bash
python -m evaluations.check_result --model_id all
```

### Output Sample

```json
{
    "time_cost": 146.17888736724854,
    "error_list": [
        1,
        2,
        2,
        2,
        2,
        2,
        0
    ],
    "cost": [
        37966,
        4465,
        0.04689600000000001,
        0
    ],
    "plan": [
        {
            "task_id": "1",
            "dependent_task_ids": [],
            "instruction": "Fine-tune the sentiment classification model using the EleutherAI/twitter-sentiment dataset",
            "task_type": "predictive modeling",
            "code": "tokenizer = GPT2Tokenizer.from_pretrained('../gpt2-small/')",
            "result": "",
            "is_success": true,
            "is_finished": true
        },
        {
            "task_id": "2",
            "dependent_task_ids": [
                "1"
            ],
            "instruction": "Verify the existence of the 'evaluation_data.parquet' file at the specified location '../' and update the file path if necessary",
            "task_type": "data exploration",
            "code": "import pandas as pd\n\n# Verify the existence of the evaluation data file\ntry:\n    evaluation_data = pd.read_parquet('../evaluation_data.parquet')\n\n    # Display basic statistical indicators\n    print(evaluation_data.describe())\n\n    # Visualize data distributions\n    import matplotlib.pyplot as plt\n    import seaborn as sns\n\n    # Histogram\n    plt.figure(figsize=(10, 6))\n    sns.histplot(evaluation_data['sentiment'], kde=False)\n    plt.title('Sentiment Distribution')\n    plt.xlabel('Sentiment')\n    plt.ylabel('Count')\n    plt.show()\n\n    # Box plot\n    plt.figure(figsize=(10, 6))\n    sns.boxplot(x=evaluation_data['sentiment'])\n    plt.title('Sentiment Distribution')\n    plt.xlabel('Sentiment')\n    plt.show()\n\n    # Correlation matrix\n    correlation_matrix = evaluation_data.corr()\n    plt.figure(figsize=(10, 6))\n    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')\n    plt.title('Correlation Matrix')\n    plt.show()\nexcept FileNotFoundError as e:\n    print('File not found:', e)",
            "result": "File not found: [Errno 2] No such file or directory: '../evaluation_data.parquet'\n",
            "is_success": true,
            "is_finished": true
        },
        {
            "task_id": "3",
            "dependent_task_ids": [
                "2"
            ],
            "instruction": "Generate a comprehensive report on the model's performance in PDF format",
            "task_type": "other",
            "code": "from reportlab.lib import colors\nfrom reportlab.lib.pagesizes import letter\nfrom reportlab.platypus import SimpleDocTemplate, Table, TableStyle\n\n# Sample data for demonstration\ndata = [['Metric', 'Value'],\n        ['Accuracy', '0.85'],\n        ['Precision', '0.78'],\n        ['Recall', '0.92'],\n        ['F1 Score', '0.84']]\n\n# Create PDF\npdf_filename = './performance_report.pdf'\npdf = SimpleDocTemplate(pdf_filename, pagesize=letter)\ntable = Table(data)\n\n# Add style to the table\nstyle = TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),\n                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),\n                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),\n                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),\n                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),\n                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),\n                    ('GRID', (0, 0), (-1, -1), 1, colors.black)])\n\ntable.setStyle(style)\n\n# Build PDF\npdf.build([table])\n\nprint(f\"Performance report generated at: {pdf_filename}\")\n",
            "result": "Performance report generated at: ./performance_report.pdf\n",
            "is_success": true,
            "is_finished": true
        }
    ]
}
```

### Evaluation

#### Evaluate functions of BigCodeBench 
```bash
python -m experiments.evaluate_tmc --model_id glm-4.5-flash
```

#### Evaluate other tasks

##### Set ground truth

Download ground truth from https://huggingface.co/datasets/zd21/DataSciBench/tree/main and move it to data/task_id/gt

##### Run evaluation
```bash
python -m experiments.evaluate

python -m experiments.evaluate --task_id all --model_id glm-4.5-flash
```

### Calculate metric

#### Calculate metrics for BigCodeBench 
```bash
python -m evaluation_results.calculate_final_metric
```
#### 


### Others

#### Modification of Data Interpreter (deprecated)

See `SciDataInterpreter.update_results_for_eval` at `role/sci_data_interpreter`. We can get the plans with codes and results from each step; the costs for each step per plan; and the number of errors. 

See `SciDataInterpreter.get_CR` at `role/sci_data_interpreter`. We can get the Completion Rate for this question that just ran. (Ground Truth not incorporated, so max at 0.5)

## **Citation**

If you find our work helpful, please kindly cite our paper:

```
@article{zhang2025datascibench,
        title={DataSciBench: An LLM Agent Benchmark for Data Science},
        author={Zhang, Dan and Zhoubian, Sining and Cai, Min and Li, Fengzu and Yang, Lekang and Wang, Wei and Dong, Tianjiao and Hu, Ziniu and Tang, Jie and Yue, Yisong},
        journal={arXiv preprint arXiv:2502.13897},
        year={2025}
        }
```
