import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Read the input CSV file
df = pd.read_csv('../combine_land.csv')

# Ensure the date_first column is of numeric type
df['date_first'] = pd.to_numeric(df['date_first'], errors='coerce')

# Filter the DataFrame based on the given conditions
filter_condition = (df['dt2IC'] == 1) & (df['dt2CIdbz'] == 1) & (df['date_first'] < 20230810000000)
filtered_df = df[filter_condition].copy()

# Features to be used for clustering
feature_columns = ['feature1', 'feature2', 'feature3']  # Replace with actual feature columns

# Prepare the data for clustering
X = filtered_df[feature_columns].values

# Initialize variables to store the best number of clusters and the highest silhouette score
best_n_clusters = 0
best_silhouette_score = -1
best_labels = None

# Perform Agglomerative Clustering for n_clusters from 2 to 10
for n_clusters in range(2, 11):
    clustering_model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = clustering_model.fit_predict(X)
    score = silhouette_score(X, labels)
    print(f'Number of clusters: {n_clusters}, Silhouette Score: {score}')
    if score > best_silhouette_score:
        best_silhouette_score = score
        best_n_clusters = n_clusters
        best_labels = labels

print(f'\nBest number of clusters: {best_n_clusters} with a Silhouette Score of {best_silhouette_score}')

# Assign the best cluster labels to the filtered DataFrame
filtered_df['clus'] = best_labels

# Merge the cluster labels back to the original DataFrame
df['clus'] = np.nan  # Initialize the 'clus' column with NaN
df.loc[filtered_df.index, 'clus'] = filtered_df['clus']

# **Subtask 1 Output:** Write the DataFrame to '1combine_land.csv' after clustering
df.to_csv('1combine_land.csv', index=False)
print("\nSubtask 1 completed: 'clus' column added and results saved to '1combine_land.csv'.")

# **Subtask 2 Processing:**

# Read the intermediate file '1combine_land.csv'
df = pd.read_csv('1combine_land.csv')

# Ensure the 'clus' column is correctly read (especially if NaNs are present)
df['clus'] = pd.to_numeric(df['clus'], errors='coerce')

# Filter the DataFrame to include only rows with cluster labels
clustered_df = df[df['clus'].notna()].copy()

# Calculate the count of each cluster
cluster_counts = clustered_df['clus'].value_counts().sort_index()
print(f'\nCluster counts:\n{cluster_counts}')

# Determine the median of all counts
median_count = int(cluster_counts.median())
print(f'\nMedian cluster count: {median_count}')

# Initialize 'keep' column with 0
df['keep'] = 0

# Assign 1 to clusters with count less than or equal to the median
clusters_to_keep_all = cluster_counts[cluster_counts <= median_count].index
indices_to_keep = clustered_df[clustered_df['clus'].isin(clusters_to_keep_all)].index
df.loc[indices_to_keep, 'keep'] = 1

# For clusters with count higher than the median, randomly select rows
clusters_to_sample = cluster_counts[cluster_counts > median_count].index
for cluster in clusters_to_sample:
    cluster_indices = clustered_df[clustered_df['clus'] == cluster].index
    sampled_indices = np.random.choice(cluster_indices, size=median_count, replace=False)
    df.loc[sampled_indices, 'keep'] = 1

# Ensure the original sorting order is maintained
df.sort_index(inplace=True)

# **Subtask 2 Output:** Write the updated DataFrame to '1combine_land.csv'
df.to_csv('1combine_land.csv', index=False)
print("\nSubtask 2 completed: 'keep' column added and results updated in '1combine_land.csv'.")

# **Final Output:** Write the completed DataFrame to 'combine_land.csv' with 'clus' and 'keep' columns
df.to_csv('combine_land.csv', index=False)
print("\nFinal output saved to 'combine_land.csv' with 'clus' and 'keep' columns.")
