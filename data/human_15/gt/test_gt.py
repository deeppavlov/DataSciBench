import pandas as pd
import matplotlib.pyplot as plt

# Create the DataFrame
data = {
    "Month": ["January", "February", "March", "January", "February", "March", "December"],
    "Year": [2022, 2022, 2022, 2023, 2023, 2023, 2023],
    "Sales": [1500, 1600, 1700, 1800, 1900, 2000, 2100],
    "Category": ["Electronics", "Electronics", "Electronics", "Furniture", "Furniture", "Furniture", "Clothing"]
}

df = pd.DataFrame(data)

# 1. Bar Chart Visualization
plt.figure(figsize=(10, 6))
total_sales_per_year = df.groupby('Year')['Sales'].sum()
total_sales_per_year.plot(kind='bar', color='red', edgecolor='black', width=0.8)
plt.xlabel('Year')
plt.ylabel('Sales')
plt.title('Total Sales per Year')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('bar_chart.png')
plt.close()

# 2. Line Chart Visualization
plt.figure(figsize=(10, 6))
df_pivot = df.pivot_table(index='Month', columns='Category', values='Sales', aggfunc='sum').reindex(["January", "February", "March", "December"])
df_pivot.plot(kind='line', style='.-.', linewidth=2)
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.title('Total Sales for Each Category by Month')
plt.legend(title='Category', loc='best')
plt.tight_layout()
plt.savefig('line_chart.png')
plt.close()

# 3. Scatter Chart Visualization
plt.figure(figsize=(10, 6))
categories = df['Category'].unique()
for category in categories:
    subset = df[df['Category'] == category]
    plt.scatter(subset['Month'] + ', ' + subset['Year'].astype(str), subset['Sales'], s=subset['Sales']/10, alpha=0.5, label=category)

plt.xlabel('Time')
plt.ylabel('Sales')
plt.title('Sales Distribution by Time and Category')
plt.legend(title='Category', loc='best')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('scatter_chart.png')
plt.close()