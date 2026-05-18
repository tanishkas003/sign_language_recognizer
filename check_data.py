# check_data.py
import pandas as pd

df = pd.read_csv("data/landmarks.csv")

print(f"Total samples : {len(df)}")
print(f"Total letters : {df['label'].nunique()}")
print(f"Columns       : {len(df.columns)} (should be 64)")
print(f"\nSamples per letter:")
print(df["label"].value_counts().sort_index())
print(f"\nAny missing values? {df.isnull().any().any()}")