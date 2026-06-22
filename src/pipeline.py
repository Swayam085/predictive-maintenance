# src/pipeline.py
# Author      : Vrushabh (Data Engineer)
# Branch      : feature/vrushabh-data
# Description : End-to-end integration pipeline
#               Raw sensor input → preprocess → alert (without ONNX)
# Week 3 Status:
#   Day 17 — integration test, batch pipeline, pipeline summary
# Used by     : Dashboard (Week 4) + Swayam (ONNX integration)

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
    print("DAY 17 — INTEGRATION TEST")
    print("="*55)

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
    print(f"\nResult:")
    print(f"  Shape    : {result['preprocessed_shape']}")
    print(f"  Prob     : {result['failure_prob']}")
    print(f"  Alert    : {result['alert_level']}")
    print(f"  Message  : {result['message']}")

    # ── Test 2: Batch pipeline ──────────────────────────
    print("\n── Test 2: Batch Pipeline (5 readings) ──────")
    batch_inputs = [
        # Normal operation
        {
            "Type": 0, "Air_temperature_K": 298.0,
            "Process_temperature_K": 308.0,
            "Rotational_speed_rpm": 1600,
            "Torque_Nm": 38.0, "Tool_wear_min": 30
        },
        # Warning zone
        {
            "Type": 1, "Air_temperature_K": 301.0,
            "Process_temperature_K": 311.0,
            "Rotational_speed_rpm": 1450,
            "Torque_Nm": 52.0, "Tool_wear_min": 130
        },
        # High risk
        {
            "Type": 2, "Air_temperature_K": 303.0,
            "Process_temperature_K": 313.0,
            "Rotational_speed_rpm": 1250,
            "Torque_Nm": 62.0, "Tool_wear_min": 200
        },
        # Critical
        {
            "Type": 2, "Air_temperature_K": 304.0,
            "Process_temperature_K": 313.5,
            "Rotational_speed_rpm": 1180,
            "Torque_Nm": 68.0, "Tool_wear_min": 230
        },
        # Normal again
        {
            "Type": 0, "Air_temperature_K": 299.0,
            "Process_temperature_K": 309.0,
            "Rotational_speed_rpm": 1550,
            "Torque_Nm": 40.0, "Tool_wear_min": 20
        }
    ]

    results = run_batch_pipeline(batch_inputs)
    print_pipeline_summary(results)
    save_pipeline_log(results)

    # ── Test 3: Alert summary ───────────────────────────
    print("\n── Test 3: Overall Alert Summary ────────────")
    get_alert_summary()
