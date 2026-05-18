# normalize_dataset.py
# Converts raw landmark coordinates into wrist-relative, scale-invariant features
# Run this ONCE on your existing data before retraining

import pandas as pd
import numpy as np

print("Loading dataset...")
df     = pd.read_csv("data/landmarks.csv")
labels = df["label"].copy()

normalized_rows = []

for idx, row in df.iterrows():
    # Wrist (landmark 0) becomes our reference point
    wx, wy, wz = row["x0"], row["y0"], row["z0"]

    # Subtract wrist from every landmark → translation invariant
    # (hand position on screen no longer matters)
    norm = []
    for i in range(21):
        norm += [
            row[f"x{i}"] - wx,
            row[f"y{i}"] - wy,
            row[f"z{i}"] - wz
        ]

    # Scale by wrist→middle-knuckle distance → scale invariant
    # (hand distance from camera no longer matters)
    ref_x = row["x9"] - wx
    ref_y = row["y9"] - wy
    scale = (ref_x**2 + ref_y**2) ** 0.5

    if scale > 0:
        norm = [v / scale for v in norm]

    normalized_rows.append(norm)

# Rebuild columns
columns = []
for i in range(21):
    columns += [f"x{i}", f"y{i}", f"z{i}"]

norm_df           = pd.DataFrame(normalized_rows, columns=columns)
norm_df["label"]  = labels.values
norm_df.to_csv("data/landmarks_normalized.csv", index=False)

print(f"Done! Saved {len(norm_df)} normalized samples.")
print(f"Letters: {sorted(norm_df['label'].unique())}")