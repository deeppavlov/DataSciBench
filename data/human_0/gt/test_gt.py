import pandas as pd
import numpy as np

def most_corr(prices):
    # Calculate daily percentage changes
    percentage_changes = prices.pct_change().dropna()

    # Initialize variables
    max_corr = 0
    pair = (None, None)

    # Get all the column names which represent tickers
    columns = prices.columns
    
    # Calculate correlation matrix
    correlation_matrix = percentage_changes.corr()

    # Iterate over the matrix looking for the highest correlation
    for i in range(len(columns)):
        for j in range(i+1, len(columns)):  # only look at upper triangle
            if correlation_matrix.iloc[i, j] > max_corr:
                max_corr = correlation_matrix.iloc[i, j]
                pair = (columns[i], columns[j])

    # Output the result to a CSV file
    output_df = pd.DataFrame([pair], columns=['Ticker 1', 'Ticker 2'])
    output_df.to_csv('most_corr_output.csv', index=False)

    return pair

# Below is the code to test the function with a given dataset
if __name__ == "__main__":
    prices = pd.read_csv("../data.csv")
    print(most_corr(prices))