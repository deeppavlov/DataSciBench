import os

import pandas as pd

# Define the file paths
source_file = '../source.xlsx'
target_file = 'target.xlsx'

# Load the source Excel file and select the 'sheet1' worksheet
source_df = pd.read_excel(source_file, sheet_name='sheet1')

# Filter the rows based on the given conditions
filtered_df = source_df[
    (source_df['D'] >= 8) & 
    (source_df['D'] < 8.0010) & 
    (source_df['F'] == 18) & 
    (source_df['G'] > source_df.loc[1, 'G'])
]

# Copy the values from the filtered rows in column G
filtered_g_values = filtered_df['G'].tolist()

# Check if the target file exists, if not, create a new DataFrame
if not os.path.exists(target_file):
    target_df = pd.DataFrame()
    with pd.ExcelWriter(target_file, engine='openpyxl') as writer:
        target_df.to_excel(writer, sheet_name='sheet1', index=False)

# Load the target Excel file and select 'sheet1' worksheet
target_df = pd.read_excel(target_file, sheet_name='sheet1')

# Make sure the column 'G' exists in the target DataFrame or create it if it doesn't
if 'G' not in target_df.columns:
    target_df['G'] = pd.NA
    
# Overwrite the corresponding cells in the target DataFrame
for i, value in enumerate(filtered_g_values):
    target_df.at[i, 'G'] = value

# Write the updated DataFrame back to the target Excel file
with pd.ExcelWriter(target_file, engine='openpyxl') as writer:
    target_df.to_excel(writer, sheet_name='sheet1', index=False)

print(f'Values from filtered rows in column G of {source_file} were successfully copied to {target_file}')