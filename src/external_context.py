# src/external_context.py
# Author      : Swayam Arya (ML Engineer)
# Branch      : feature/swayam-ml
# Week 2      : Simulate external context (ambient temp, load density)
#               and merge with IoT sensor data via precise timestamps

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Paths
CLEAN_DATA_PATH  = os.path.join("data", "processed", "clean_data.csv")
FUSED_DATA_PATH  = os.path.join("data", "processed", "fused_data.csv")


def generate_timestamps(n_rows: int, start: str = "2026-01-01 00:00:00",
                        freq_minutes: int = 10) -> pd.Series:
    """
    Generate a precise timestamp for each IoT sensor reading.
    Readings simulated every `freq_minutes` minutes.
    """
    start_dt = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
    timestamps = [start_dt + timedelta(minutes=freq_minutes * i) for i in range(n_rows)]
    return pd.Series(timestamps)


def generate_external_context(n_rows: int, seed: int = 42) -> pd.DataFrame:
    """
    Simulate external context data with precise timestamps:
    - timestamp      : exact time of reading
    - ambient_temp    : factory ambient temperature (Celsius)
    - load_density    : factory load percentage (0 to 1)
    - humidity        : humidity percentage
    - shift           : work shift (morning / afternoon / night)
    """
    np.random.seed(seed)

    timestamps   = generate_timestamps(n_rows)
    ambient_temp = np.random.normal(loc=28.0, scale=4.0, size=n_rows)
    load_density = np.clip(np.random.normal(loc=0.65, scale=0.15, size=n_rows), 0.1, 1.0)
    humidity     = np.clip(np.random.normal(loc=55.0, scale=10.0, size=n_rows), 20.0, 90.0)

    shifts = []
    for ts in timestamps:
        hour = ts.hour
        if 6 <= hour < 14:
            shifts.append("morning")
        elif 14 <= hour < 22:
            shifts.append("afternoon")
        else:
            shifts.append("night")

    df_ext = pd.DataFrame({
        "timestamp"    : timestamps,
        "ambient_temp" : np.round(ambient_temp, 2),
        "load_density" : np.round(load_density, 3),
        "humidity"     : np.round(humidity, 2),
        "shift"        : shifts,
    })

    print(f"[EXTERNAL] Generated {n_rows} rows of external context.")
    print(f"  Timestamp range → {timestamps.iloc[0]} to {timestamps.iloc[-1]}")
    print(f"  Ambient Temp    → mean: {ambient_temp.mean():.2f} std: {ambient_temp.std():.2f}")
    print(f"  Load Density    → mean: {load_density.mean():.2f} std: {load_density.std():.2f}")
    print(f"  Humidity        → mean: {humidity.mean():.2f} std: {humidity.std():.2f}")

    return df_ext


def encode_shift(df: pd.DataFrame) -> pd.DataFrame:
    """Encode shift column into numbers: morning=0, afternoon=1, night=2"""
    shift_map = {"morning": 0, "afternoon": 1, "night": 2}
    df["shift_encoded"] = df["shift"].map(shift_map)
    df = df.drop(columns=["shift"])
    return df


def attach_timestamps_to_iot(df_iot: pd.DataFrame, timestamps: pd.Series) -> pd.DataFrame:
    """
    Attach the same precise timestamp sequence to IoT sensor readings,
    simulating that each row was logged at that exact moment.
    """
    df_iot = df_iot.reset_index(drop=True).copy()
    df_iot["timestamp"] = timestamps.values
    return df_iot


def merge_with_iot(df_iot: pd.DataFrame, df_ext: pd.DataFrame) -> pd.DataFrame:
    """
    Merge IoT sensor data with external context using an exact
    timestamp join (pd.merge on 'timestamp' column) — this is the
    precise-timestamp merge required by the project spec.
    """
    df_iot = df_iot.reset_index(drop=True)
    df_ext = df_ext.reset_index(drop=True)

    df_fused = pd.merge(df_iot, df_ext, on="timestamp", how="inner")

    print(f"\n[MERGE] IoT shape     : {df_iot.shape}")
    print(f"[MERGE] External shape: {df_ext.shape}")
    print(f"[MERGE] Fused shape   : {df_fused.shape}  (merged on 'timestamp')")

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
    print("  WEEK 2 — EXTERNAL CONTEXT FUSION (Timestamp-based)")
    print("  Branch: feature/swayam-ml")
    print("="*50)

    df_iot = pd.read_csv(CLEAN_DATA_PATH)
    print(f"\n[IOT] Clean data loaded: {df_iot.shape}")

    df_ext = generate_external_context(n_rows=len(df_iot))

    timestamps = df_ext["timestamp"]
    df_iot = attach_timestamps_to_iot(df_iot, timestamps)

    df_ext = encode_shift(df_ext)

    df_fused = merge_with_iot(df_iot, df_ext)

    df_fused = add_interaction_features(df_fused)

    df_fused_ml = df_fused.drop(columns=["timestamp"])

    save_fused_data(df_fused_ml)

    print("\n[DONE] External context fusion (timestamp-based) complete.")
    print("[NEXT] Use fused_data.csv in model.py for improved CV")