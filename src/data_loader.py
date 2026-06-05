import pandas as pd
import numpy as np

def load_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Dataset loaded! Shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    return df

def basic_info(df):
    print("\n--- Dataset Info ---")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nFailure count:\n{df['Machine failure'].value_counts()}")

if __name__ == "__main__":
    df = load_data("data/raw/ai4i2020.csv")
    basic_info(df)