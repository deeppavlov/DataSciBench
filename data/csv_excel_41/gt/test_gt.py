import pandas as pd

# Read the data
df = pd.read_csv('../data.csv')

# Calculate the proportion of missing values in each column
missing_proportion = df.isnull().mean()


# Create a mask for rows with at least two non-missing values in the last three columns
mask = df[['weight', 'age', 'salary']].notnull().sum(axis=1) >= 2

# Select the rows based on the mask
selected_rows = df[mask]


# Save the selected rows to 'missing_values_proportion.csv'
selected_rows.to_csv('missing_values_proportion.csv', index=False)


# Convert the missing proportions to a DataFrame
missing_proportion_df = missing_proportion.to_frame(name='missing_proportion').reset_index()
missing_proportion_df.rename(columns={'index': 'column'}, inplace=True)

# Save the missing value proportions
missing_proportion_df.to_csv('missing_values_proportion.csv', index=False)


import numpy as np
import pandas as pd

# Read the data
df = pd.read_csv('../data.csv')


def fill_weight(row, models):
    if pd.isnull(row['weight']):
        region = row['region']
        height = row['height']
        model = models[region] if region in models else models['overall']
        # Predict weight using the model
        predicted_weight = model['slope'] * height + model['intercept']
        return predicted_weight
    else:
        return row['weight']


# Initialize a dictionary to store models
models = {}

# Build models for each region
regions = df['region'].unique()
for region in regions:
    df_region = df[(df['region'] == region) & df['weight'].notnull() & df['height'].notnull()]
    if len(df_region) >= 2:
        slope, intercept = np.polyfit(df_region['height'], df_region['weight'], 1)
        models[region] = {'slope': slope, 'intercept': intercept}

# Build an overall model using all data
df_all = df[df['weight'].notnull() & df['height'].notnull()]
if len(df_all) >= 2:
    slope, intercept = np.polyfit(df_all['height'], df_all['weight'], 1)
    models['overall'] = {'slope': slope, 'intercept': intercept}


# Fill missing weights
df['weight'] = df.apply(lambda row: fill_weight(row, models), axis=1)


# Save the updated dataset
df.to_csv('updated_data.csv', index=False)
