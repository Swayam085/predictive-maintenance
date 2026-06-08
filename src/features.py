import pandas as pd
import numpy as np
import os

PROCESSED_PATH = os.path.join("data", "processed", "clean_data.csv")
FEATURED_PATH  = os.path.join("data", "processed", "featured_data.csv")


def load_clean(path: str = PROCESSED_PATH) -> pd.DataFrame:

    df = pd.read_csv(path)
    print(f"[INFO] Loaded clean data: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"[INFO] Columns: {df.columns.tolist()}")
    return df


def add_power_feature(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df["Power"] = df["Torque_Nm"] * df["Rotational_speed_rpm"]
    print(f"[FEAT] Power added → min: {df['Power'].min():.2f}, "
          f"max: {df['Power'].max():.2f}, "
          f"mean: {df['Power'].mean():.2f}")
    return df


def add_temp_diff_feature(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df["Temp_diff"] = df["Process_temperature_K"] - df["Air_temperature_K"]
    print(f"[FEAT] Temp_diff added → min: {df['Temp_diff'].min():.2f}, "
          f"max: {df['Temp_diff'].max():.2f}, "
          f"mean: {df['Temp_diff'].mean():.2f}")
    return df


def add_wear_bins(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    bins   = [0, 100, 200, float("inf")]
    labels = [0, 1, 2]   # Low=0, Medium=1, High=2
    df["Tool_wear_bin"] = pd.cut(
        df["Tool_wear_min"],
        bins=bins,
        labels=labels,
        include_lowest=True
    ).astype(int)

    print(f"[FEAT] Tool_wear_bin distribution:\n"
          f"       Low(0)   : {(df['Tool_wear_bin']==0).sum()}\n"
          f"       Medium(1): {(df['Tool_wear_bin']==1).sum()}\n"
          f"       High(2)  : {(df['Tool_wear_bin']==2).sum()}")
    return df


def add_interaction_terms(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["Torque_x_Wear"] = df["Torque_Nm"] * df["Tool_wear_min"]
    print(f"[FEAT] Torque_x_Wear added → mean: {df['Torque_x_Wear'].mean():.2f}")

    df["Power_x_Temp"] = df["Power"] * df["Temp_diff"]
    print(f"[FEAT] Power_x_Temp added  → mean: {df['Power_x_Temp'].mean():.2f}")

    return df


def add_outlier_flags(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    
    check_cols = ["Torque_Nm", "Rotational_speed_rpm", "Tool_wear_min", "Power"]

    for col in check_cols:
        Q1  = df[col].quantile(0.25)   # 25th percentile
        Q3  = df[col].quantile(0.75)   # 75th percentile
        IQR = Q3 - Q1                  # Interquartile Range

        lower = Q1 - 1.5 * IQR        # Lower bound
        upper = Q3 + 1.5 * IQR        # Upper bound

        flag_col = f"{col}_outlier"
        df[flag_col] = ((df[col] < lower) | (df[col] > upper)).astype(int)

        outlier_count = df[flag_col].sum()
        print(f"[FEAT] {flag_col}: {outlier_count} outliers "
              f"(range: {lower:.2f} – {upper:.2f})")

    return df


def save_featured(df: pd.DataFrame, path: str = FEATURED_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\n[SAVE]  Saved → {path}")
    print(f"[SAVE] Final shape: {df.shape}")
    print(f"[SAVE] All columns: {df.columns.tolist()}")


def add_rpm_per_torque(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df["RPM_per_Torque"] = df["Rotational_speed_rpm"] / (df["Torque_Nm"] + 1e-6)
    print(f"[FEAT] RPM_per_Torque  -> mean: {df['RPM_per_Torque'].mean():.4f}")
    return df


def add_wear_rate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Wear_rate"] = df["Tool_wear_min"] / (df["Rotational_speed_rpm"] + 1e-6)
    print(f"[FEAT] Wear_rate       -> mean: {df['Wear_rate'].mean():.6f}")
    return df


def add_multi_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    flag_cols = [
        "Torque_Nm_outlier",
        "Rotational_speed_rpm_outlier",
        "Tool_wear_min_outlier",
        "Power_outlier"
    ]
    df["Multi_flag"] = df[flag_cols].sum(axis=1)
    print(f"[FEAT] Multi_flag dist -> {df['Multi_flag'].value_counts().sort_index().to_dict()}")
    print(f"[FEAT] Critical (3-4)  -> {(df['Multi_flag'] >= 3).sum()} rows")
    return df

if __name__ == "__main__":
    df = load_clean()

    # Day 2 features
    df = add_power_feature(df)
    df = add_temp_diff_feature(df)
    df = add_wear_bins(df)
    df = add_interaction_terms(df)
    df = add_outlier_flags(df)

    # Day 3 features
    df = add_rpm_per_torque(df)
    df = add_wear_rate(df)
    df = add_multi_flag(df)

    save_featured(df)