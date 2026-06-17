# src/model.py
# Author      : Swayam Arya (ML Engineer)
# Branch      : feature/swayam-ml
# Week 2      : 5-Fold CV + Ablation Study + Final Model Train
# Week 3 TODO : SMOTE inside CV folds, threshold tuning

import os
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report
)
from imblearn.over_sampling import SMOTE

# Paths
FEATURED_PATH = os.path.join("data", "processed", "featured_data.csv")
X_TRAIN_PATH  = os.path.join("data", "processed", "X_train.csv")
Y_TRAIN_PATH  = os.path.join("data", "processed", "y_train.csv")
X_TEST_PATH   = os.path.join("data", "processed", "X_test.csv")
Y_TEST_PATH   = os.path.join("data", "processed", "y_test.csv")

# LightGBM Params
LGBM_PARAMS = {
    "n_estimators"  : 200,
    "learning_rate" : 0.05,
    "max_depth"     : 6,
    "num_leaves"    : 31,
    "class_weight"  : "balanced",
    "random_state"  : 42,
    "n_jobs"        : -1,
    "verbose"       : -1,
}

def train_model(X_train, y_train):
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("\n" + "=" * 50)
    print("LIGHTGBM MODEL EVALUATION")
    print("=" * 50)
    print(classification_report(y_test, y_pred))
    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision : {precision_score(y_test, y_pred):.3f}")
    print(f"Recall    : {recall_score(y_test, y_pred):.3f}")
    print(f"F1 Score  : {f1_score(y_test, y_pred):.3f}")
    return y_pred

def show_feature_importance(model, feature_names):
    importance = model.feature_importances_
    ranked = sorted(
        zip(feature_names, importance),
        key=lambda x: x[1],
        reverse=True
    )
    print("\nTop Feature Importance")
    print("-" * 40)
    for feature, score in ranked:
        print(f"  {feature:<30} {score}")

def get_feature_sets(df):
    baseline_cols = [
        "Air_temperature_K",
        "Process_temperature_K",
        "Rotational_speed_rpm",
        "Torque_Nm",
        "Tool_wear_min",
    ]
    external_cols = [
        "Power",
        "Temp_diff",
        "Tool_wear_bin",
        "Torque_x_Wear",
        "Power_x_Temp",
        "RPM_per_Torque",
        "Wear_rate",
        "Multi_flag",
    ]
    base = [c for c in baseline_cols if c in df.columns]
    ext  = [c for c in external_cols  if c in df.columns]
    return {
        "baseline"     : base,
        "with_external": base + ext,
    }

def run_cv(X, y, use_smote=False, label="Model"):
    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    print(f"\n{'='*50}")
    print(f"CV — {label}  |  SMOTE: {use_smote}")
    print(f"{'='*50}")
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        if use_smote:
            sm = SMOTE(random_state=42, k_neighbors=5)
            X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
        model  = train_model(X_tr, y_tr)
        y_pred = model.predict(X_val)
        f1_mac = f1_score(y_val, y_pred, average="macro")
        scores.append(f1_mac)
        print(f"  Fold {fold} → Macro F1: {f1_mac:.4f}")
    mean_f1 = np.mean(scores)
    std_f1  = np.std(scores)
    print(f"\n  Mean Macro F1 : {mean_f1:.4f} ± {std_f1:.4f}")
    return {
        "label"  : label,
        "mean_f1": round(mean_f1, 4),
        "std_f1" : round(std_f1, 4),
        "scores" : scores,
    }

def run_ablation_study(df, target_col="Machine_failure"):
    feature_sets = get_feature_sets(df)
    y = df[target_col]
    results = []
    for name, cols in feature_sets.items():
        print(f"\n[ABLATION] {name} — {len(cols)} features")
        X = df[cols]
        result = run_cv(X, y, use_smote=False, label=name)
        results.append(result)
    print(f"\n{'='*50}")
    print("  ABLATION RESULTS")
    print(f"{'='*50}")
    print(f"  {'Setup':<22} {'Mean F1':>8}  {'Std':>6}")
    print(f"  {'-'*38}")
    for r in results:
        print(f"  {r['label']:<22} {r['mean_f1']:>8.4f}  {r['std_f1']:>6.4f}")
    improvement = results[1]["mean_f1"] - results[0]["mean_f1"]
    print(f"\n  External features improved Macro F1 by: +{improvement:.4f}")
    os.makedirs("reports", exist_ok=True)
    df_res = pd.DataFrame(results)[["label", "mean_f1", "std_f1"]]
    df_res.to_csv("reports/ablation_results.csv", index=False)
    print("[SAVE] reports/ablation_results.csv")
    return df_res

def train_final_model(X_train, y_train, use_smote=False):
    if use_smote:
        sm = SMOTE(random_state=42, k_neighbors=5)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print(f"[SMOTE] Resampled → 0:{(y_train==0).sum()} 1:{(y_train==1).sum()}")
    model = train_model(X_train, y_train)
    print(f"[TRAIN] Final model trained on {len(X_train)} samples.")
    return model

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  WEEK 2 — ML PIPELINE  (feature/swayam-ml)")
    print("="*50)

    df = pd.read_csv(FEATURED_PATH)
    print(f"[INFO] Data loaded: {df.shape}")

    ablation_df = run_ablation_study(df)

    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()

    feature_sets  = get_feature_sets(df)
    best_features = [f for f in feature_sets["with_external"] if f in X_train.columns]

    print(f"\n[TRAIN] Using {len(best_features)} features")
    final_model = train_final_model(X_train[best_features], y_train)

    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).squeeze()
    X_test = X_test[[f for f in best_features if f in X_test.columns]]

    evaluate_model(final_model, X_test, y_test)
    show_feature_importance(final_model, best_features)

    print("\n[DONE] Week 2 complete.")
    print("[NEXT] Week 3: SMOTE inside CV + threshold tuning")