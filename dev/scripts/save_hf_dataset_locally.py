import os
import json
from datasets import load_dataset

SOURCE_DATASET="tbd"  # Update this line to match the actual dataset.
TARGET_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "..", # project root
    "datasets", SOURCE_DATASET)
TRAIN_PATH = os.path.join(TARGET_DIR, "train.jsonl")
TEST_PATH = os.path.join(TARGET_DIR, "test.jsonl")

def main():

    # Check target directory
    if os.path.exists(TARGET_DIR) is False:
        os.makedirs(TARGET_DIR)
        print(f"Created {TARGET_DIR}")

    # Load the dataset
    dataset = load_dataset(SOURCE_DATASET)

    # Save train split
    with open(TRAIN_PATH, "w") as f:
        for item in dataset["train"]:
            f.write(json.dumps(item) + "\n")

    # Save test split
    with open(TEST_PATH, "w") as f:
        for item in dataset["test"]:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()