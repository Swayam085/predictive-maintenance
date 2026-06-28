# dashboard/data_feed.py
# Author      : Vrushabh (Data Engineer)
# Branch      : feature/vrushabh-data
# Description : Live sensor data stream simulator
#               CSV replay → real-time sensor readings for dashboard
# Week 4 Status:
#   Day 22 — data_feed, stream_sensor_data, get_single_reading
# Used by     : dashboard/app.py (Streamlit)

import pandas as pd
import numpy as np
import time
import os
import sys

sys.path.append("src")
from inference import preprocess_input, load_feature_cols
from alert_system import process_alert, get_alert_summary

# ── Paths ──────────────────────────────────────────────
FEATURED_CSV = os.path.join("data", "processed", "featured_data.csv")

# ── Raw sensor columns needed ──────────────────────────
RAW_COLS = [
    "Type", "Air_temperature_K", "Process_temperature_K",
    "Rotational_speed_rpm", "Torque_Nm", "Tool_wear_min"
]


def load_sensor_data(path: str = FEATURED_CSV) -> pd.DataFrame:
  
    df = pd.read_csv(path)

    # Sirf raw sensor columns rakho
    missing = [c for c in RAW_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[RAW_COLS + ["Machine_failure"]].copy()
    print(f"[FEED] Loaded {len(df)} sensor readings")
    return df


def get_single_reading(df: pd.DataFrame,
                        idx: int = None) -> dict:

    if idx is None:
        idx = np.random.randint(0, len(df))

    row = df.iloc[idx]
    reading = {
        "idx"                   : int(idx),
        "Type"                  : int(row["Type"]),
        "Air_temperature_K"     : float(row["Air_temperature_K"]),
        "Process_temperature_K" : float(row["Process_temperature_K"]),
        "Rotational_speed_rpm"  : int(row["Rotational_speed_rpm"]),
        "Torque_Nm"             : float(row["Torque_Nm"]),
        "Tool_wear_min"         : int(row["Tool_wear_min"]),
        "actual_failure"        : int(row["Machine_failure"])
    }
    return reading


def simulate_stream(df: pd.DataFrame,
                    n_readings: int = 20,
                    delay: float = 0.5) -> list:

    print(f"\n[STREAM] Starting live simulation — {n_readings} readings")
    print("="*55)

    feature_cols = load_feature_cols()
    results      = []

    for i in range(n_readings):
        # Random reading lo
        reading = get_single_reading(df)

        # Sensor input (actual_failure aur idx remove karo)
        sensor_input = {k: v for k, v in reading.items()
                        if k not in ["idx", "actual_failure"]}

        # Preprocess karo
        X = preprocess_input(sensor_input, feature_cols)

        # Simulated probability — tool wear + torque based
        wear_ratio   = reading["Tool_wear_min"] / 250.0
        torque_ratio = reading["Torque_Nm"] / 80.0
        temp_ratio   = (reading["Process_temperature_K"] -
                        reading["Air_temperature_K"]) / 15.0
        prob = min(0.99, (wear_ratio * 0.5 +
                          torque_ratio * 0.3 +
                          temp_ratio * 0.2))

        # Alert process karo
        pred_label = 1 if prob >= 0.30 else 0
        alert      = process_alert(
            sensor_input, prob, pred_label,
            save_history=False   # Dashboard mein history save mat karo
        )

        result = {
            "reading_num"    : i + 1,
            "idx"            : reading["idx"],
            "sensor_input"   : sensor_input,
            "failure_prob"   : round(prob, 4),
            "alert_level"    : alert["alert_level"],
            "message"        : alert["message"],
            "actual_failure" : reading["actual_failure"],
            "timestamp"      : alert["timestamp"]
        }
        results.append(result)

        # Print karo
        print(f"[{i+1:02d}] Alert: {result['alert_level']:6s} | "
              f"Prob: {prob:.4f} | "
              f"Actual: {reading['actual_failure']} | "
              f"Tool wear: {reading['Tool_wear_min']}min")

        # Delay (dashboard mein use hoga)
        if delay > 0:
            time.sleep(delay)

    return results


def get_stream_summary(results: list) -> dict:

    total  = len(results)
    green  = sum(1 for r in results if r["alert_level"] == "GREEN")
    yellow = sum(1 for r in results if r["alert_level"] == "YELLOW")
    red    = sum(1 for r in results if r["alert_level"] == "RED")
    actual_failures = sum(1 for r in results if r["actual_failure"] == 1)

    # Accuracy
    correct = sum(
        1 for r in results
        if (r["alert_level"] in ["YELLOW", "RED"]) == bool(r["actual_failure"])
    )

    summary = {
        "total"           : total,
        "GREEN"           : green,
        "YELLOW"          : yellow,
        "RED"             : red,
        "actual_failures" : actual_failures,
        "correct"         : correct,
        "accuracy"        : round(correct / total * 100, 1)
    }

    print(f"\n{'='*55}")
    print(f"STREAM SUMMARY")
    print(f"{'='*55}")
    print(f"Total readings   : {total}")
    print(f"GREEN  (safe)    : {green}  ({green/total*100:.1f}%)")
    print(f"YELLOW (warn)    : {yellow}  ({yellow/total*100:.1f}%)")
    print(f"RED    (crit)    : {red}  ({red/total*100:.1f}%)")
    print(f"Actual failures  : {actual_failures}")
    print(f"Accuracy         : {summary['accuracy']}%")
    print(f"{'='*55}")

    return summary


# ── Direct run ─────────────────────────────────────────
if __name__ == "__main__":
    print("="*55)
    print("DAY 22 — DATA FEED TEST")
    print("="*55)

    # Step 1: Data load karo
    df = load_sensor_data()
    print(f"[FEED] Columns: {df.columns.tolist()}")
    print(f"[FEED] Shape  : {df.shape}")

    # Step 2: Single reading test
    print("\n── Single Reading Test ───────────────────────")
    reading = get_single_reading(df, idx=0)
    print(f"Reading: {reading}")

    # Step 3: Stream simulate karo (10 readings, no delay for test)
    print("\n── Stream Simulation (10 readings) ──────────")
    results = simulate_stream(df, n_readings=10, delay=0)

    # Step 4: Summary
    summary = get_stream_summary(results)

    print("\n[INFO] data_feed.py — Day 22 verified!")