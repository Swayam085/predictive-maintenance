# src/external_context.py
# Author      : Swayam Arya (ML Engineer)
# Branch      : feature/swayam-ml
# Week 2      : Simulate external context (ambient temp, load density)
#               and merge with IoT sensor data via timestamps

import os
import numpy as np
import pandas as pd

# Paths
CLEAN_DATA_PATH  = os.path.join("data", "processed", "clean_data.csv")
FUSED_DATA_PATH  = os.path.join("data", "processed", "fused_data.csv")


def generate_external_context(n_rows: int, seed: int = 42) -> pd.DataFrame:
    """
    Simulate external context data:
    - ambient_temp     : factory ambient temperature (Celsius)
    - load_density     : factory load percentage (0 to 1)
    - shift            : work shift (morning / afternoon / night)
    - humidity         : humidity percentage
    """
    np.random.seed(seed)

    ambient_temp = np.random.normal(loc=28.0, scale=4.0, size=n_rows)
    load_density = np.clip(np.random.normal(loc=0.65, scale=0.15, size=n_rows), 0.1, 1.0)
    humidity     = np.clip(np.random.normal(loc=55.0, scale=10.0, size=n_rows), 20.0, 90.0)

    shifts = []
    for i in range(n_rows):
        if i % 3 == 0:
            shifts.append("morning")
        elif i % 3 == 1:
            shifts.append("afternoon")
        else:
            shifts.append("night")

    df_ext = pd.DataFrame({
        "ambient_temp" : np.round(ambient_temp, 2),
        "load_density" : np.round(load_density, 3),
        "humidity"     : np.round(humidity, 2),
        "shift"        : shifts,
    })

    print(f"[EXTERNAL] Generated {n_rows} rows of external context.")
    print(f"  Ambient Temp  → mean: {ambient_temp.mean():.2f} std: {ambient_temp.std():.2f}")
    print(f"  Load Density  → mean: {load_density.mean():.2f} std: {load_density.std():.2f}")
    print(f"  Humidity      → mean: {humidity.mean():.2f} std: {humidity.std():.2f}")

    return df_ext


def encode_shift(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode shift column into numbers:
    morning = 0, afternoon = 1, night = 2
    """
    shift_map = {"morning": 0, "afternoon": 1, "night": 2}
    df["shift_encoded"] = df["shift"].map(shift_map)
    df = df.drop(columns=["shift"])
    return df


def merge_with_iot(df_iot: pd.DataFrame,
                   df_ext: pd.DataFrame) -> pd.DataFrame:
    """
    Merge IoT sensor data with external context.
    Both have same number of rows — merge by index (timestamp simulation).
    """
    df_ext = df_ext.reset_index(drop=True)
    df_iot = df_iot.reset_index(drop=True)

    df_fused = pd.concat([df_iot, df_ext], axis=1)

    print(f"\n[MERGE] IoT shape     : {df_iot.shape}")
    print(f"[MERGE] External shape: {df_ext.shape}")
    print(f"[MERGE] Fused shape   : {df_fused.shape}")

    return df_fused


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Week 2 — new interaction features using external context:
    - temp_load_stress : ambient_temp x load_density
    - humidity_wear    : humidity x Tool_wear_min
    - night_load       : is night shift AND high load
    """
    df["temp_load_stress"] = np.round(df["ambient_temp"] * df["load_density"], 3)

    if "Tool_wear_min" in df.columns:
        df["humidity_wear"] = np.round(df["humidity"] * df["Tool_wear_min"], 3)

    if "shift_encoded" in df.columns:
        df["night_load"] = (
            (df["shift_encoded"] == 2) & (df["load_density"] > 0.7)
        ).astype(int)

    print(f"\n[FEATURES] New interaction features added:")
    print(f"  temp_load_stress, humidity_wear, night_load")

    return df


def save_fused_data(df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(FUSED_DATA_PATH), exist_ok=True)
    df.to_csv(FUSED_DATA_PATH, index=False)
    print(f"\n[SAVE] Fused data saved → {FUSED_DATA_PATH}")
    print(f"[INFO] Shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")


if __name__ == "__main__":
    print("\n" + "="*50)
    print("  WEEK 2 — EXTERNAL CONTEXT FUSION")
    print("  Branch: feature/swayam-ml")
    print("="*50)

    # Load IoT clean data
    df_iot = pd.read_csv(CLEAN_DATA_PATH)
    print(f"\n[IOT] Clean data loaded: {df_iot.shape}")

    # Generate external context
    df_ext = generate_external_context(n_rows=len(df_iot))

    # Encode shift
    df_ext = encode_shift(df_ext)

    # Merge
    df_fused = merge_with_iot(df_iot, df_ext)

    # Add interaction features
    df_fused = add_interaction_features(df_fused)

    # Save
    save_fused_data(df_fused)

    print("\n[DONE] External context fusion complete.")
    print("[NEXT] Use fused_data.csv in model.py for improved CV")