# src/inference.py
# Author      : Vrushabh (Data Engineer)
# Branch      : feature/vrushabh-data
# Description : ONNX inference pipeline
#               Raw sensor input → preprocessing → ONNX model → failure probability
# Week 3 Status:
#   Day 15 — inference pipeline, preprocess, predict functions
# Used by     : Swayam (ONNX export) + Dashboard (Week 4)

import pandas as pd
import numpy as np
import os
import json

# ── Paths ──────────────────────────────────────────────
ONNX_MODEL_PATH  = os.path.join("models", "model.onnx")
FEATURE_COLS_PATH = os.path.join("models", "feature_cols.json")


def load_feature_cols(path: str = FEATURE_COLS_PATH) -> list:
  
    if not os.path.exists(path):
        # Default feature columns — Day 2+3 ke features
        feature_cols = [
            "Type", "Air_temperature_K", "Process_temperature_K",
            "Rotational_speed_rpm", "Torque_Nm", "Tool_wear_min",
            "Power", "Temp_diff", "Tool_wear_bin",
            "Torque_x_Wear", "Power_x_Temp",
            "Torque_Nm_outlier", "Rotational_speed_rpm_outlier",
            "Tool_wear_min_outlier", "Power_outlier",
            "RPM_per_Torque", "Wear_rate", "Multi_flag"
        ]
        print(f"[INFO] feature_cols.json not found — using default 18 features")
        return feature_cols

    with open(path, "r") as f:
        feature_cols = json.load(f)
    print(f"[INFO] Feature cols loaded: {len(feature_cols)} features")
    return feature_cols


def preprocess_input(raw_input: dict,
                     feature_cols: list) -> np.ndarray:
   
    df = pd.DataFrame([raw_input])

    # Feature 1: Power
    df["Power"] = df["Torque_Nm"] * df["Rotational_speed_rpm"]

    # Feature 2: Temp_diff
    df["Temp_diff"] = df["Process_temperature_K"] - df["Air_temperature_K"]

    # Feature 3: Tool_wear_bin
    bins   = [0, 100, 200, float("inf")]
    labels = [0, 1, 2]
    df["Tool_wear_bin"] = pd.cut(
        df["Tool_wear_min"],
        bins=bins,
        labels=labels,
        include_lowest=True
    ).astype(int)

    # Feature 4: Interaction terms
    df["Torque_x_Wear"] = df["Torque_Nm"] * df["Tool_wear_min"]
    df["Power_x_Temp"]  = df["Power"] * df["Temp_diff"]

    # Feature 5: Efficiency ratios
    df["RPM_per_Torque"] = df["Rotational_speed_rpm"] / (df["Torque_Nm"] + 1e-6)
    df["Wear_rate"]      = df["Tool_wear_min"] / (df["Rotational_speed_rpm"] + 1e-6)

    # Feature 6: IQR outlier flags
    # Hardcoded bounds from training data (Day 2 output)
    outlier_bounds = {
        "Torque_Nm"           : (12.80, 67.20),
        "Rotational_speed_rpm": (1139.50, 1895.50),
        "Tool_wear_min"       : (-110.50, 325.50),
        "Power"               : (32452.88, 87526.27)
    }
    for col, (lower, upper) in outlier_bounds.items():
        flag_col     = f"{col}_outlier"
        df[flag_col] = ((df[col] < lower) | (df[col] > upper)).astype(int)

    # Feature 7: Multi_flag
    flag_cols       = [
        "Torque_Nm_outlier",
        "Rotational_speed_rpm_outlier",
        "Tool_wear_min_outlier",
        "Power_outlier"
    ]
    df["Multi_flag"] = df[flag_cols].sum(axis=1)

    # Missing columns ko 0 se fill karo
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
            print(f"[WARN] Column '{col}' missing — filled with 0")

    df = df[feature_cols]

    print(f"[PREPROCESS] Input shape: {df.shape}")
    print(f"[PREPROCESS] Features   : {df.columns.tolist()}")

    # float32 — ONNX requirement
    return df.values.astype(np.float32)


def load_onnx_model(path: str = ONNX_MODEL_PATH):
   
    import onnxruntime as rt

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[ERROR] ONNX model not found at '{path}'\n"
            f"        Swayam ka model export pehle run karo (Week 3 Day 16)"
        )

    session = rt.InferenceSession(path)
    print(f"[ONNX] Model loaded from → {path}")
    print(f"[ONNX] Input  : {session.get_inputs()[0].name} "
          f"| shape: {session.get_inputs()[0].shape}")
    print(f"[ONNX] Output : {session.get_outputs()[0].name}")
    return session


def predict(session, X: np.ndarray) -> dict:
    input_name  = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Prediction run karo
    outputs = session.run([output_name], {input_name: X})
    pred_label = int(outputs[0][0])

    # Probability nikalo (agar available ho)
    try:
        prob_output = session.get_outputs()[1].name
        probs       = session.run([prob_output], {input_name: X})
        failure_prob = float(probs[0][0][1])   # class 1 probability
    except Exception:
        failure_prob = float(pred_label)

    result = {
        "pred_label"  : pred_label,
        "failure_prob": round(failure_prob, 4),
        "alert"       : get_alert_level(failure_prob)
    }

    print(f"[PREDICT] Label       : {pred_label} "
          f"({'FAILURE' if pred_label == 1 else 'NO FAILURE'})")
    print(f"[PREDICT] Probability : {failure_prob:.4f}")
    print(f"[PREDICT] Alert       : {result['alert']}")

    return result


def get_alert_level(prob: float) -> str:
  
    if prob < 0.30:
        return "GREEN"
    elif prob < 0.60:
        return "YELLOW"
    else:
        return "RED"


def run_inference(raw_input: dict,
                  model_path: str = ONNX_MODEL_PATH) -> dict:
    feature_cols = load_feature_cols()
    X            = preprocess_input(raw_input, feature_cols)
    session      = load_onnx_model(model_path)
    result       = predict(session, X)
    return result


if __name__ == "__main__":

    # Sample raw sensor input — 1 machine reading
    sample_input = {
        "Type"                  : 1,       # M = 1
        "Air_temperature_K"     : 300.0,
        "Process_temperature_K" : 310.5,
        "Rotational_speed_rpm"  : 1500,
        "Torque_Nm"             : 45.0,
        "Tool_wear_min"         : 150
    }

    print("="*50)
    print("INFERENCE PIPELINE TEST")
    print("="*50)
    print(f"Input: {sample_input}")
    print()

    # Step 1: Feature cols load karo
    feature_cols = load_feature_cols()
    print(f"\n[INFO] Total features: {len(feature_cols)}")

    # Step 2: Preprocess karo
    X = preprocess_input(sample_input, feature_cols)
    print(f"\n[INFO] Preprocessed shape : {X.shape}")
    print(f"[INFO] Preprocessed values: {X[0][:6]}...")

    # Step 3: Alert level test karo (ONNX model ke bina)
    print("\n── Alert Level Tests ────────────────────────")
    for prob in [0.10, 0.35, 0.45, 0.65, 0.90]:
        alert = get_alert_level(prob)
        print(f"Prob: {prob:.2f} → Alert: {alert}")

    print("\n[INFO] inference.py — preprocessing verified!")