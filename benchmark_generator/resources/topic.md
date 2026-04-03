# Data Science Tasks

Generate simple tasks related to data analysis, machine learning, statistical analysis, and data visualization.

Tasks may include:
- Building predictive models on tabular data
- Exploratory data analysis and reporting
- Training and evaluating classifiers/regressors
- Time series analysis
- Clustering and dimensionality reduction

Open-ended tasks: analytical reports, written conclusions, textual summaries of findings.
Tasks without input files: use built-in datasets from libraries (sklearn, etc.).
Tasks with input files: generate synthetic datasets via input_data.py.

## External Storage (filesystem MCP)
If the `filesystem` MCP server is available, it represents an **External Storage**.
- **Allowed Path:** Only paths starting with `/tmp/bench_fs/` are accessible via the filesystem tools (`read_file`, `write_file`, etc.).
- **Local vs External:** Standard Python (pandas, open()) MUST be used for all local files in the current working directory.
- **Integration Tasks:** Generate some tasks focused on data archiving, backups, or moving results from the local context to the External Storage at `/tmp/bench_fs/`.
