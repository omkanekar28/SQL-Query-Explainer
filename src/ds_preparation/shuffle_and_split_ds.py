import os
import pandas as pd
from src.common import DatasetSettings

INPUT_DATASET_PATH = "data/dataset_incomplete.csv"
RANDOM_SEED = DatasetSettings.RANDOM_SEED

TRAIN_SPLIT_SIZE = DatasetSettings.TRAIN_SPLIT_SIZE
VAL_SPLIT_SIZE = DatasetSettings.VAL_SPLIT_SIZE
TEST_SPLIT_SIZE = DatasetSettings.TEST_SPLIT_SIZE

OUTPUT_DATASET_DIR = "data/splits"

if TRAIN_SPLIT_SIZE + VAL_SPLIT_SIZE + TEST_SPLIT_SIZE != 1.0:
    raise ValueError("Total split size must be 1.0!")

# Load dataset
df = pd.read_csv(INPUT_DATASET_PATH)

# Shuffle dataset
df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

# Calculate split indices
total_rows = len(df)
train_end = int(total_rows * TRAIN_SPLIT_SIZE)
val_end = train_end + int(total_rows * VAL_SPLIT_SIZE)

# Split dataset
train_df = df.iloc[:train_end]
val_df = df.iloc[train_end:val_end]
test_df = df.iloc[val_end:]

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DATASET_DIR, exist_ok=True)

# Save splits
train_df.to_csv(os.path.join(OUTPUT_DATASET_DIR, "train.csv"), index=False)
val_df.to_csv(os.path.join(OUTPUT_DATASET_DIR, "validation.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_DATASET_DIR, "test.csv"), index=False)

print(f"Dataset split complete!")
print(f"Train: {len(train_df)} rows")
print(f"Validation: {len(val_df)} rows")
print(f"Test: {len(test_df)} rows")
print(f"Saved to: {OUTPUT_DATASET_DIR}")