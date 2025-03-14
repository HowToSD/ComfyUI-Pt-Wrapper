"""
Train val split by HowToSD
"""
import os
import sys
import pandas as pd

def main():
    target_file_path = os.path.join(os.path.dirname(__file__), "bezdekIris_with_header.data")

    df = pd.read_csv(target_file_path)

    # Split to 3 dfs
    df1 = df.query('`class`=="Iris-setosa"')
    df2 = df.query('`class`=="Iris-versicolor"')
    df3 = df.query('`class`=="Iris-virginica"')
    assert len(df1) == 50
    assert len(df2) == 50
    assert len(df3) == 50
    
    train_indices = []
    val_indices = []

    # Train/validation split indices (20% validation)
    val_indices = list(range(0, 50, 5))  # [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    train_indices = list(set(range(50)) - set(val_indices))

    # Apply train/val split
    df1_train, df1_val = df1.iloc[train_indices], df1.iloc[val_indices]
    df2_train, df2_val = df2.iloc[train_indices], df2.iloc[val_indices]
    df3_train, df3_val = df3.iloc[train_indices], df3.iloc[val_indices]

    # Concatenate all train and val datasets
    df_train = pd.concat([df1_train, df2_train, df3_train], ignore_index=True)
    df_val = pd.concat([df1_val, df2_val, df3_val], ignore_index=True)

    # Check dataset sizes
    assert len(df_train) == 120
    assert len(df_val) == 30

    df_train.to_csv(os.path.join(os.path.dirname(__file__), "iris_train.csv"))
    df_val.to_csv(os.path.join(os.path.dirname(__file__), "iris_val.csv"))

if __name__ == "__main__":
    main()