# %% [markdown]
# ### Pass@k

# %%
import os
import csv
import pandas as pd

base_path = "/path/to/DataSciBench/evaluation_results/"
model_list = [
                'o1-mini', 
                "gpt-4o-2024-05-13", 'gpt-4o-mini','gpt-4-turbo', 'claude-3-5-sonnet-20240620', "glm-4-flash", #"claude-3-opus-20240229", 
                "meta-llama/Meta-Llama-3.1-8B-Instruct", "meta-llama/Meta-Llama-3-8B-Instruct", 
                "google/gemma-2-9b-it", "THUDM/glm-4-9b-chat", "Qwen/Qwen2.5-7B-Instruct", 
                "Qwen/Qwen2-7B-Instruct", "Qwen/Qwen2-1.5B-Instruct", "01-ai/Yi-1.5-9B-Chat-16K",
                "CodeLlama-34b-Instruct-hf", "CodeLlama-13b-Instruct-hf", "CodeLlama-7b-Instruct-hf", "starcoder2-15b", 
                "starcoder2-7b", "starcoder2-3b", 
                "deepseek-coder-33b-instruct", "deepseek-coder-6.7b-instruct",  "deepseek-coder-1.3b-instruct",  
                "Qwen/Qwen2.5-Coder-7B-Instruct", 
                "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                "Meta-Llama-3.1-70B-Instruct", 
                ]
# write the header to the final csv file
df = pd.DataFrame({
    "Model": ["Model"],
    "Pass@1": ["Pass@1"],
    # "Pass@1 (Fail)": ["Pass@1 (Fail)"],
    "Average CR": ["Average CR"],
    "VLM": ["VLM"],
    "F1": ["F1"],
    "F2": ["F2"],
    "F3": ["F3"],
    "F4": ["F4"],
    "F5": ["F5"],
    "Pass@1 (Human)": ["Pass@1 (Human)"],
    "Pass@1 (DL)": ["Pass@1 (DL)"],
    "Pass@1 (CSV)": ["Pass@1 (CSV)"],
    "Avg. CR (Human)": ["Avg. CR (Human)"],
    "Avg. CR (DL)": ["Avg. CR (DL)"],
    "Avg. CR (CSV)": ["Avg. CR (CSV)"],
})
csv_output_file = base_path + "final_results.csv"
with open(csv_output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(df.columns)
# write this header to the final csv file, just replace the file if it already exists
# df.to_csv(base_path + "final_results.csv", index=False)
# df.to_csv(base_path + "final_results.csv", mode='w', header=True, index=False)

TOTAL_DATA_NUM = 55
HUMAN_DATA_NUM = 25
CSV_DATA_NUM = 20
DL_DATA_NUM = 10

TOTAL_NUM_FUNC = 167
HUMAN_NUM_FUNC = 82
CSV_NUM_FUNC = 58
DL_NUM_FUNC = 27

F1_NUM_FUNC = 28
F2_NUM_FUNC = 1
F3_NUM_FUNC = 10
F4_NUM_FUNC = 8
F5_NUM_FUNC = 19

F_NUM_LIST = [F1_NUM_FUNC, F2_NUM_FUNC, F3_NUM_FUNC, F4_NUM_FUNC, F5_NUM_FUNC]

for model in model_list:
    model = model.split("/")[-1]
    if not os.path.exists(base_path + model + "_results.csv"):
        print(f"skipping {model}")
        continue
    print('-'*100)
    print(f"Model: {model}")
    df = pd.read_csv(base_path + model + "_results.csv")

    # %%
    # get all the row where result_type is Completion Rate

    df_crs = df[df['result_type'] == 'Completion Rate']

    # %%
    success_dict = {}
    for index, row in df_crs.iterrows():
        if row['data_name'] not in success_dict:
            success_dict[row['data_name']] = 0
        if row['result_cr'] == 1:
            success_dict[row['data_name']] += 1


    # %%
    # count the number of success and total
    success_counter = 0
    fail_counter = 0
    total_counter = 0
    for key in success_dict:
        total_counter += 1
        success_counter += success_dict[key] / 10
        # if success_dict[key] == "Success":
        #     success_counter += 1
        # if success_dict[key] == "Fail":
        #     fail_counter += 1

    total_pass = success_counter/TOTAL_DATA_NUM * 100
    total_fail = fail_counter/TOTAL_DATA_NUM
    print("Pass@1: ", total_pass)
    print("Pass@1 (Fail): ", fail_counter/TOTAL_DATA_NUM)


    # %%
    # count the number of success and total for data_name starting with "human"
    success_counter = 0
    total_counter = 0
    for key in success_dict:
        if key.startswith("human"):
            total_counter += 1
            success_counter += success_dict[key] / 10
            # if success_dict[key] == "Success":
            #     success_counter += 1

    human_pass = success_counter/HUMAN_DATA_NUM * 100
    print("Pass@1 for data_name starting with 'human': ", human_pass)

    # %%
    # count the number of success and total for data_name starting with "csv"
    success_counter = 0
    total_counter = 0
    for key in success_dict:
        if key.startswith("csv"):
            total_counter += 1
            success_counter += success_dict[key] / 10
            # if success_dict[key] == "Success":
            #     success_counter += 1

    csv_pass = success_counter/CSV_DATA_NUM * 100
    print("Pass@1 for data_name starting with 'csv': ", csv_pass)

    # %%
    # count the number of success and total for data_name starting with "dl"
    success_counter = 0
    total_counter = 0
    for key in success_dict:
        if key.startswith("dl"):
            total_counter += 1
            if success_dict[key] == "Success":
                success_counter += 1

    dl_pass = success_counter/DL_DATA_NUM * 100
    print("Pass@1 for data_name starting with 'dl': ", dl_pass)

    # %%
    # calculate average completion rate for df_crs
    total_cr = 0
    for index, row in df_crs.iterrows():
        total_cr += row['result_cr']

    avg_cr = total_cr/(TOTAL_NUM_FUNC * 10) * 100
    print("Average completion rate: ", avg_cr)

    # calculate average completion rate for human, csv and dl
    total_cr = 0
    for index, row in df_crs.iterrows():
        if row['data_name'].startswith("human"):
            total_cr += row['result_cr']

    avg_human_cr = total_cr/(HUMAN_NUM_FUNC * 10) * 100
    print("Average completion rate for data_name starting with 'human': ", total_cr/(HUMAN_NUM_FUNC * 10) * 100)

    total_cr = 0
    for index, row in df_crs.iterrows():
        if row['data_name'].startswith("csv"):
            total_cr += row['result_cr']

    avg_csv_cr = total_cr/(CSV_NUM_FUNC * 10) * 100
    print("Average completion rate for data_name starting with 'csv': ", total_cr/(CSV_NUM_FUNC * 10) * 100)

    total_cr = 0
    for index, row in df_crs.iterrows():
        if row['data_name'].startswith("dl"):
            total_cr += row['result_cr']
    
    avg_dl_cr = total_cr/(DL_NUM_FUNC * 10) * 100
    print("Average completion rate for data_name starting with 'dl': ", total_cr/(DL_NUM_FUNC * 10) * 100)
    # %% [markdown]
    # ## Calcualte CR of Top-k functions

    # %%
    top_5_functions_dict = [
        {"function_name": "Data Quality Score", "task_name": "Data cleaning and preprocessing"},
        {"function_name": "Plot Validity", "task_name": "Data visualization"},
        {"function_name": "Data Accuracy", "task_name": "Data exploration and statistics understand"},
        {"function_name": "Visualization Completeness", "task_name": "Data visualization"},
        {"function_name": "Model Accuracy", "task_name": "Predictive modeling"}
    ]

    # %%
    # calculate average completion rate for each (funciotn, task) in top_5_functions_dict
    f_list = []
    for function_dict, TOTAL in zip(top_5_functions_dict, F_NUM_LIST):
        # print("function_name: ", function_dict['function_name'], "task_name: ", function_dict['task_name'])
        total_cr = 0
        for index, row in df.iterrows():
            # print(row['function_name'])
            try:
                if row['function_name'] == function_dict['function_name'] and row['task_name'] == function_dict['task_name']:
                    total_cr += row['result_cr']
            except:
                print(row)
        try:
            f_list.append(total_cr/(TOTAL * 2 * 10) * 100)
            print("Average completion rate for (", function_dict['function_name'], ", ", function_dict['task_name'], "): ", total_cr/(TOTAL * 2 * 10))
        except:
            f_list.append(0)
            print("Average completion rate for (", function_dict['function_name'], ", ", function_dict['task_name'], "): ", 0)
    print('-'*100)
    # %%
    vlm_list = []
    for index, row in df.iterrows():
        if row['result_type'] == "single task (int)":
            vlm_list.append(int(row['result_value']))

    try:
        vlm_score = sum(vlm_list)/len(vlm_list)
    except:
        vlm_score = 0
    print("VLM: ", vlm_score)

    # store all the above metircs to a csv file by model name. All float should be converted to .4f
    df = pd.DataFrame({
        "Model": [model],
        "Pass@1": [f"{total_pass:.2f}"],
        # "Pass@1 (Fail)": [f"{total_fail:.2f}"],
        "Average CR": [f"{avg_cr:.2f}"],
        "VLM": [f"{vlm_score:.2f}"],
        "F1": [f"{f_list[0]:.2f}"],
        "F2": [f"{f_list[1]:.2f}"],
        "F3": [f"{f_list[2]:.2f}"],
        "F4": [f"{f_list[3]:.2f}"],
        "F5": [f"{f_list[4]:.2f}"],
        "Pass@1 (Human)": [f"{human_pass:.2f}"],
        "Pass@1 (DL)": [f"{dl_pass:.2f}"],
        "Pass@1 (CSV)": [f"{csv_pass:.2f}"],
        "Avg. CR (Human)": [f"{avg_human_cr:.2f}"],
        "Avg. CR (DL)": [f"{avg_dl_cr:.2f}"],
        "Avg. CR (CSV)": [f"{avg_csv_cr:.2f}"],
    })
    # append this to the final csv file
    df.to_csv(base_path + "final_results.csv", mode='a', header=False, index=False)


# %%
