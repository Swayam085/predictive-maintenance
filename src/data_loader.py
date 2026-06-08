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
        print(" There is no missing value!")

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

    # 5. Failure sub-types drop 
    failure_subtypes = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    drop_sub = [c for c in failure_subtypes if c in df.columns]
    if drop_sub:
        df = df.drop(columns=drop_sub)
        print(f"[CLEAN] Dropped failure sub-types: {drop_sub}")

    print(f"\n[CLEAN]  Final shape: {df.shape}")
    return df


def save_processed(df: pd.DataFrame, path: str = PROCESSED_PATH) -> None:
   
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[SAVE]  Saved → {path}")

def make_splits(df: pd.DataFrame,
                target_col: str = "Machine_failure",
                test_size: float = 0.15,
                val_size: float = 0.15,
                random_state: int = 42) -> dict:
   
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=[target_col])
    y = df[target_col]

    print(f"\n[SPLIT] Total samples  : {len(df)}")
    print(f"[SPLIT] Features       : {X.shape[1]} columns")
    print(f"[SPLIT] Target         : {target_col}")
    print(f"[SPLIT] Failure rate   : {y.mean()*100:.2f}%")

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    val_ratio = val_size / (1 - test_size)   # 15/85 = 0.176
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        stratify=y_temp,
        random_state=random_state
    )

    for name, X_s, y_s in [
        ("Train", X_train, y_train),
        ("Val  ", X_val,   y_val),
        ("Test ", X_test,  y_test)
    ]:
        print(f"[SPLIT] {name} → {len(X_s):5d} rows | "
              f"failure rate: {y_s.mean()*100:.2f}%")


    os.makedirs(os.path.join("data", "processed"), exist_ok=True)

    X_train.to_csv(os.path.join("data","processed","X_train.csv"), index=False)
    X_val.to_csv  (os.path.join("data","processed","X_val.csv"),   index=False)
    X_test.to_csv (os.path.join("data","processed","X_test.csv"),  index=False)
    y_train.to_csv(os.path.join("data","processed","y_train.csv"), index=False)
    y_val.to_csv  (os.path.join("data","processed","y_val.csv"),   index=False)
    y_test.to_csv (os.path.join("data","processed","y_test.csv"),  index=False)

    print(f"\n[SAVE] 6 files saved → data/processed/")
    print(f"[SAVE] X_train, X_val, X_test, y_train, y_val, y_test")

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test
    }



if __name__ == "__main__":
    # Original pipeline (Day 1)
    df_raw   = load_raw()
    inspect(df_raw)
    df_clean = clean(df_raw)
    save_processed(df_clean)

    # Day 4 — featured data on split 
    print("\n" + "="*50)
    print("SPLIT PIPELINE")
    print("="*50)
    df_featured = pd.read_csv(
        os.path.join("data", "processed", "featured_data.csv")
    )
    splits = make_splits(df_featured)
