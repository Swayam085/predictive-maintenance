# src/data_loader.py

import pandas as pd
import os

# ─ Paths
RAW_PATH       = os.path.join("data", "raw", "ai4i2020.csv")
PROCESSED_PATH = os.path.join("data", "processed", "clean_data.csv")


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[INFO] Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def inspect(df: pd.DataFrame) -> None:
  

    print("\n── COLUMNS ──────────────────────────────────────────")
    print(df.columns.tolist())

    print("\n── DTYPES ───────────────────────────────────────────")
    print(df.dtypes)

    print("\n── MISSING VALUES ───────────────────────────────────")
    missing = df.isnull().sum()
    if missing.any():
        print(missing[missing > 0])
    else:
        print(" Koi missing value nahi!")

    print("\n── DUPLICATE ROWS ───────────────────────────────────")
    dupes = df.duplicated().sum()
    print(f"Duplicate rows: {dupes}")

    print("\n── BASIC STATS ──────────────────────────────────────")
    print(df.describe())

    print("\n── TARGET BALANCE (Machine failure) ─────────────────")
    vc = df["Machine failure"].value_counts()
    print(vc)
    pct = df["Machine failure"].mean() * 100
    print(f"\nFailure rate: {pct:.2f}%  ← class imbalance check")


def clean(df: pd.DataFrame) -> pd.DataFrame:

    # 1. Column names clean
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"[\s\[\]()]+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )
    print("\n[CLEAN] Renamed columns:", df.columns.tolist())

    # 2. Duplicates remove
    before = len(df)
    df = df.drop_duplicates()
    print(f"[CLEAN] Duplicate rows removed: {before - len(df)}")

    # 3. Identifier columns drop
    drop_ids = [c for c in ["UDI", "Product_ID"] if c in df.columns]
    df = df.drop(columns=drop_ids)
    print(f"[CLEAN] Dropped identifier columns: {drop_ids}")

    # 4. Type encode (ordinal: L < M < H)
    type_map = {"L": 0, "M": 1, "H": 2}
    if "Type" in df.columns:
        df["Type"] = df["Type"].map(type_map)
        print("[CLEAN] 'Type' encoded → L:0, M:1, H:2")

    # 5. Failure sub-types drop (target leakage prevent)
    failure_subtypes = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    drop_sub = [c for c in failure_subtypes if c in df.columns]
    if drop_sub:
        df = df.drop(columns=drop_sub)
        print(f"[CLEAN] Dropped failure sub-types: {drop_sub}")

    print(f"\n[CLEAN]  Final shape: {df.shape}")
    return df


def save_processed(df: pd.DataFrame, path: str = PROCESSED_PATH) -> None:
    """Clean data ko processed/ folder mein save karo."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[SAVE]  Saved → {path}")

if __name__ == "__main__":
    df_raw   = load_raw()
    inspect(df_raw)
    df_clean = clean(df_raw)
    save_processed(df_clean)
