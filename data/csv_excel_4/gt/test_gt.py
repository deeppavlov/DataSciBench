import matplotlib.pyplot as plt
import pandas as pd


def detect_outliers(data_series, window_size, threshold):
    outliers = []
    means = data_series.rolling(window=window_size, center=True).mean()
    stds = data_series.rolling(window=window_size, center=True).std()
    
    for i in range(len(data_series)):
        if i < window_size // 2 or i >= len(data_series) - window_size // 2:
            outliers.append(False)
        else:
            mean = means[i]
            std = stds[i]
            value = data_series[i]
            if pd.isna(mean) or pd.isna(std):
                outliers.append(False)
            elif abs(value - mean) > threshold * std:
                outliers.append(True)
            else:
                outliers.append(False)
    return outliers

def replace_outliers(data_series, outliers):
    smoothed_data = data_series.copy()
    for i in range(len(data_series)):
        if outliers[i]:
            valid_data = data_series[max(0, i-5) : min(len(data_series), i+6)]
            smoothed_data[i] = valid_data.mean()
    return smoothed_data

# Load data
data = pd.read_csv("../ultimate_data.csv")
closing_prices = data["closing price"]

# Detect outliers
outlier_flags = detect_outliers(closing_prices, window_size=15, threshold=3)

# Export detected outliers
outliers_df = pd.DataFrame({
    "index": data.index,
    "value": closing_prices,
    "is_outlier": outlier_flags
})
outliers_df.to_csv("outliers_detected.csv", index=False)

# Replace outliers and export smoothed data
smoothed_prices = replace_outliers(closing_prices, outlier_flags)
smoothed_data_df = pd.DataFrame({
    "index": data.index,
    "smoothed_closing_price": smoothed_prices
})
smoothed_data_df.to_csv("smoothed_data.csv", index=False)

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(closing_prices, label='Original Closing Prices', color='blue', alpha=0.5)
plt.plot(smoothed_prices, label='Smoothed Closing Prices', color='red')
plt.scatter(data.index, closing_prices, c=outlier_flags, cmap='coolwarm', label='Outliers', alpha=0.7)
plt.title("Closing Price: Outliers Detection and Replacement")
plt.xlabel("Index")
plt.ylabel("Closing Price")
plt.legend()
plt.grid(True)
plt.savefig("outlier_detection_plot.png")
plt.show()