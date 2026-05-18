# train_model.py
# This script:
#   1. Loads your landmark dataset
#   2. Splits it into training and testing sets
#   3. Trains a Random Forest classifier
#   4. Evaluates accuracy
#   5. Saves the trained model to models/

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

print("=" * 50)
print("        ASL Model Training")
print("=" * 50)

# ── Step 1: Load data ──────────────────────────────
df = pd.read_csv("data/landmarks_normalized.csv")
print(f"\nDataset loaded: {len(df)} samples, {df['label'].nunique()} letters")

# Separate features (the 63 numbers) from the label (the letter)
# X = input features, y = target labels
# This naming convention (X, y) is universal in ML — you'll see it everywhere
X = df.drop("label", axis=1).values   # shape: (total_samples, 63)
y = df["label"].values                 # shape: (total_samples,)

print(f"Feature matrix shape : {X.shape}")
print(f"Labels shape         : {y.shape}")

# ── Step 2: Train/Test Split ───────────────────────
# We split data into 80% training, 20% testing
# The model NEVER sees test data during training
# test_size=0.2 means 20% goes to testing
# random_state=42 makes the split reproducible (same split every run)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# stratify=y ensures each letter is proportionally represented in both splits

print(f"\nTraining samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

# ── Step 3: Train the model ────────────────────────
print("\nTraining Random Forest... ", end="", flush=True)

model = RandomForestClassifier(
    n_estimators=100,    # build 100 decision trees
    max_depth=20,        # each tree can be at most 20 levels deep
    random_state=42      # reproducibility
)

model.fit(X_train, y_train)   # THIS is where learning happens
print("Done!")

# ── Step 4: Evaluate accuracy ──────────────────────
y_pred = model.predict(X_test)   # predict on data model has never seen

accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# Detailed per-letter breakdown
print("\nPer-letter performance:")
print(classification_report(y_test, y_pred))

# ── Step 5: Confusion Matrix ───────────────────────
# A confusion matrix shows which letters get confused with each other
# Perfect model = bright diagonal line, everything else is dark
labels = sorted(df["label"].unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(14, 12))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)
plt.title("Confusion Matrix — ASL Letter Recognition", fontsize=14)
plt.xlabel("Predicted Letter")
plt.ylabel("Actual Letter")
plt.tight_layout()
plt.savefig("models/confusion_matrix.png", dpi=150)
plt.show()
print("\nConfusion matrix saved to models/confusion_matrix.png")

# ── Step 6: Save the trained model ────────────────
# pickle serializes the model object to a file
# so we can load and use it later without retraining
model_path = "models/asl_model_normalized.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"Model saved to {model_path}")
print("\nTraining complete!")