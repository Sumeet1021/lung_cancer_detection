import pandas as pd

data = pd.read_csv("dataset/lung_cancer.csv")

print("Columns:")
print(data.columns)

print("\nShape:")
print(data.shape)