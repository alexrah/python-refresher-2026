import pandas as pd
import numpy as np

is_printable = False

np.random.seed(42)

data = {
    'name': [f'Person_{i}' for i in range(1, 51)],
    'age': np.random.randint(18, 65, 50),
    'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose'], 50),
    'salary': np.random.randint(30000, 120000, 50)
}

basic_df: pd.DataFrame = pd.DataFrame(data)

if is_printable:
    print("First 5 rows:")
    print(basic_df.head())
    print("\nLast 5 rows:")
    print(basic_df.tail())
    print("\nDataFrame info:")
    print(basic_df.info())
    print("\nDescriptive statistics:")
    print(basic_df.describe())
    print("\nMean age:")
    print(basic_df['age'].mean())
    print("\nMedian age:")
    print(basic_df['age'].median())
    print("\nManual median calculation:")

row_count = len(basic_df)
sorted_ages = basic_df.sort_values('age')['age']
if row_count % 2 == 0:
    median = (sorted_ages.iloc[row_count // 2] + sorted_ages.iloc[row_count // 2 - 1]) / 2
else:
    median = sorted_ages.iloc[row_count // 2]
print(f"Manual median calculation: {median}")

if is_printable:
    print(basic_df.sort_values('age').iloc[24])
    print("\nStandard deviation of age:")
    print(basic_df['age'].std())






