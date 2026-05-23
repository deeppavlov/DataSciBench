import pandas as pd

# Data provided as input
data = [
    (2022, "农业科技进步贡献率", 62.4),
    (2021, "农业科技进步贡献率", 61.5),
    (2020, "化肥有效利用率", 40.2),
    (2020, "农药有效利用率", 40.6),
    (2020, "县域数字农业农村发展总体水平", 37.9),
    (2020, "农业科技进步贡献率", 60.7),
    (2019, "农药有效利用率", 39.8),
    (2019, "化肥有效利用率", 39.2),
    (2019, "县域数字农业农村发展总体水平", 36.0),
    (2019, "农业科技进步贡献率", 59.2),
    (2018, "县域数字农业农村发展总体水平", 33.0),
    (2018, "农业科技进步贡献率", 58.3),
    (2017, "农药有效利用率", 38.8),
    (2017, "化肥有效利用率", 38.8),
    (2017, "农业科技进步贡献率", 57.5),
    (2016, "农业科技进步贡献率", 56.7),
    (2015, "农药有效利用率", 36.6),
    (2015, "化肥有效利用率", 35.2),
    (2015, "农业科技进步贡献率", 56.1),
    (2014, "农业科技进步贡献率", 55.6),
    (2013, "农药有效利用率", 35.0),
    (2013, "化肥有效利用率", 33.0),
    (2013, "农业科技进步贡献率", 55.2),
    (2012, "农业科技进步贡献率", 54.5)
]

# Transform data into a dictionary suitable for DataFrame
data_dict = {}
for year, category, value in data:
    if year not in data_dict:
        data_dict[year] = {}
    data_dict[year][category] = value

# Create DataFrame and transpose it to switch rows and columns
df = pd.DataFrame(data_dict).transpose()
df.index.name = "Year"

# Save DataFrame to output.csv
df.to_csv("output.csv")

# Printing DataFrame to console for verification
print(df)