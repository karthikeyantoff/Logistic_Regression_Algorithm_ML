import pandas as pd
df=pd.read_csv('D:\DATA_SCIENCE_PROJECT\DATA_SETS\housing.csv')
print(df.head())
import numpy as np
url = r"D:\DATA_SCIENCE_PROJECT\DATA_SETS\housing.csv"
print("Downloading Raw Data...")
df = pd.read_csv(url)

df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].mean())
# df['total_bedrooms']=df['total_bedrooms'].fillna(df[])

mapping = {'<1H OCEAN': 0, 'INLAND': 1, 'NEAR OCEAN': 2, 'NEAR BAY': 3, 'ISLAND': 4}
df['ocean_proximity'] = df['ocean_proximity'].map(mapping)

df.to_csv('housing_cleaned.csv', index=False)
print("Data Cleaned & Saved as 'housing_cleaned.csv'")
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# df = pd.read_csv('D:\DATA_SCIENCE_PROJECT\DATA_SETS\housing_cleaned.csv')
# plt.figure(figsize=(10, 6))
# numeric_df = df.select_dtypes(include=['float64', 'int64'])
# sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
# plt.title("Correlation Heatmap: What affects House Prices?")
# plt.show()
# plt.figure(figsize=(8, 5))
# sns.scatterplot(x='median_income', y='median_house_value', data=df, alpha=0.3)
# plt.title("Income vs. House Price")
# plt.xlabel("Median Income (Scaled)")
# plt.ylabel("House Value ($)")
# plt.show()
# plt.figure(figsize=(8, 5))
# sns.histplot(df['median_house_value'], bins=50, kde=True, color='green')
# plt.title("Distribution of House Prices")
# plt.xlabel("Price ($)")
# plt.show()