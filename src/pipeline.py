# src/pipeline.py
# Author      : Vrushabh (Data Engineer)
# Branch      : feature/vrushabh-data
# Description : End-to-end integration pipeline
#               Raw input / CSV → preprocess → alert → log
# Week 3 Status:
#   Day 17 — run_pipeline_without_onnx, run_batch_pipeline
#             save_pipeline_log, print_pipeline_summary
#   Day 18 — run_pipeline_from_csv, compare_alerts_with_actual
#             CSV test: 90% accuracy on 10 rows
#   Day 19 — 27/27 pytest tests passing
# Used by     : Dashboard (Week 4)

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

# ── Local imports ───────────────────────────────────────
from inference import preprocess_input, load_feature_cols, get_alert_level
from alert_system import process_alert, get_alert_summary


# ── Paths ───────────────────────────────────────────────
PIPELINE_LOG = os.path.join("reports", "pipeline_log.json")


def run_pipeline_without_onnx(sensor_input: dict,
                               prob: float = None) -> dict:
  
    feature_cols = load_feature_cols()
    X            = preprocess_input(sensor_input, feature_cols)

    # Agar prob nahi diya toh simulate karo
    if prob is None:
        # Tool wear aur Torque se rough estimate
        wear_ratio = sensor_input.get("Tool_wear_min", 0) / 250.0
        torque_ratio = sensor_input.get("Torque_Nm", 40) / 80.0
        prob = min(0.99, (wear_ratio + torque_ratio) / 2)

    pred_label = 1 if prob >= 0.30 else 0
    alert      = process_alert(sensor_input, prob, pred_label)

    result = {
        "sensor_input"     : sensor_input,
        "preprocessed_shape": X.shape,
        "failure_prob"     : round(prob, 4),
        "pred_label"       : pred_label,
        "alert_level"      : alert["alert_level"],
        "message"          : alert["message"],
        "timestamp"        : alert["timestamp"]
    }

    return result


def run_batch_pipeline(batch_inputs: list) -> list:
  
    print(f"\n[BATCH] Processing {len(batch_inputs)} sensor readings...")
    print("="*55)

    results = []
    for i, sensor_input in enumerate(batch_inputs):
        print(f"\n[BATCH] Reading {i+1}/{len(batch_inputs)}")
        result = run_pipeline_without_onnx(sensor_input)
        results.append(result)
        print(f"[BATCH] → Alert: {result['alert_level']} "
              f"| Prob: {result['failure_prob']:.4f}")

    return results


def save_pipeline_log(results: list,
                      path: str = PIPELINE_LOG) -> None:

    os.makedirs(os.path.dirname(path), exist_ok=True)

    log = {
        "timestamp" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_runs": len(results),
        "results"   : results
    }

    with open(path, "w") as f:
        json.dump(log, f, indent=4)
    print(f"\n[LOG] Pipeline log saved → {path}")

def run_pipeline_from_csv(csv_path: str,
                           n_samples: int = 10) -> list:
    """
    CSV file se sensor readings load karo aur batch pipeline run karo.
    Real data pe end-to-end test karne ke liye.
    Args:
        csv_path  : path to CSV file (featured_data.csv)
        n_samples : number of rows to process (default 10)
    Returns:
        list of result dicts
    """
    print(f"\n[CSV] Loading data from → {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"[CSV] Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    # Raw sensor columns jo inference.py ko chahiye
    raw_cols = [
        "Type", "Air_temperature_K", "Process_temperature_K",
        "Rotational_speed_rpm", "Torque_Nm", "Tool_wear_min"
    ]

    # Missing columns check karo
    missing = [c for c in raw_cols if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing columns: {missing}")
        return []

    # Sample rows lo
    sample_df = df[raw_cols].head(n_samples)
    print(f"[CSV] Processing {n_samples} rows...")

    # Har row ko dict mein convert karo
    batch_inputs = sample_df.to_dict(orient="records")

    # Batch pipeline run karo
    results = run_batch_pipeline(batch_inputs)
    return results


def compare_alerts_with_actual(csv_path: str,
                                results: list,
                                n_samples: int = 10) -> dict:

    df = pd.read_csv(csv_path)

    if "Machine_failure" not in df.columns:
        print("[ERROR] Machine_failure column not found!")
        return {}

    actual = df["Machine_failure"].head(n_samples).tolist()

    print("\n" + "="*55)
    print("ALERT vs ACTUAL COMPARISON")
    print("="*55)
    print(f"{'Row':<5} {'Actual':<10} {'Alert':<10} {'Prob':<8} {'Match'}")
    print("-"*55)

    correct = 0
    for i, (res, act) in enumerate(zip(results, actual)):
        alert     = res["alert_level"]
        prob      = res["failure_prob"]
        # RED/YELLOW = predicted failure, GREEN = predicted no failure
        pred_fail = 1 if alert in ["RED", "YELLOW"] else 0
        match     = "YES" if pred_fail == act else "NO"
        if match == "YES":
            correct += 1
        print(f"{i+1:<5} {act:<10} {alert:<10} {prob:<8.4f} {match}")

    accuracy = correct / n_samples * 100
    print("-"*55)
    print(f"Correct: {correct}/{n_samples} ({accuracy:.1f}%)")
    print("="*55)

    return {
        "total"   : n_samples,
        "correct" : correct,
        "accuracy": round(accuracy, 2)
    }


def print_pipeline_summary(results: list) -> None:
  
    total  = len(results)
    green  = sum(1 for r in results if r["alert_level"] == "GREEN")
    yellow = sum(1 for r in results if r["alert_level"] == "YELLOW")
    red    = sum(1 for r in results if r["alert_level"] == "RED")

    print("\n" + "="*55)
    print("PIPELINE SUMMARY")
    print("="*55)
    print(f"Total readings : {total}")
    print(f"GREEN  (safe)  : {green}  ({green/total*100:.1f}%)")
    print(f"YELLOW (warn)  : {yellow}  ({yellow/total*100:.1f}%)")
    print(f"RED    (crit)  : {red}  ({red/total*100:.1f}%)")
    print("="*55)


# ── Direct run ─────────────────────────────────────────
if __name__ == "__main__":

    print("="*55)
    print("DAY 18 — CSV INPUT + COMPARISON TEST")
    print("="*55)

    CSV_PATH = os.path.join("data", "processed", "featured_data.csv")

    # ── Test 1: Single reading ──────────────────────────
    print("\n── Test 1: Single Sensor Reading ────────────")
    single_input = {
        "Type"                  : 1,
        "Air_temperature_K"     : 300.0,
        "Process_temperature_K" : 310.5,
        "Rotational_speed_rpm"  : 1500,
        "Torque_Nm"             : 45.0,
        "Tool_wear_min"         : 50
    }
    result = run_pipeline_without_onnx(single_input)
    print(f"  Shape  : {result['preprocessed_shape']}")
    print(f"  Prob   : {result['failure_prob']}")
    print(f"  Alert  : {result['alert_level']}")

    # ── Test 2: CSV pipeline ────────────────────────────
    print("\n── Test 2: CSV Pipeline (10 rows) ───────────")
    csv_results = run_pipeline_from_csv(CSV_PATH, n_samples=10)
    print_pipeline_summary(csv_results)
    save_pipeline_log(csv_results)

    # ── Test 3: Compare with actual ─────────────────────
    print("\n── Test 3: Alert vs Actual Comparison ───────")
    stats = compare_alerts_with_actual(CSV_PATH, csv_results, n_samples=10)

    # ── Test 4: Alert summary ───────────────────────────
    print("\n── Test 4: Overall Alert Summary ────────────")
    get_alert_summary()

    print("\n[INFO] pipeline.py — Day 18 CSV test complete!")