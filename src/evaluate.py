# src/evaluate.py
# Author      : Vrushabh (Data Engineer)
# Branch      : feature/vrushabh-data
# Description : Model evaluation metrics module
#               F1, Precision, Recall, ROC-AUC, Confusion Matrix
# Used by     : Swayam (model.py) for LightGBM evaluation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

FIGURES_PATH = os.path.join("reports", "figures")


def evaluate_model(y_true, y_pred, y_prob,
                   model_name: str = "Model") -> dict:
   
    f1        = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall    = recall_score(y_true, y_pred)
    auc       = roc_auc_score(y_true, y_prob)

    print(f"\n{'='*50}")
    print(f"EVALUATION — {model_name}")
    print(f"{'='*50}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"ROC-AUC   : {auc:.4f}")
    print(f"\n── Classification Report ────────────────────")
    print(classification_report(y_true, y_pred,
                                target_names=["No Failure", "Failure"]))

    return {
        "model"    : model_name,
        "f1"       : round(f1, 4),
        "precision": round(precision, 4),
        "recall"   : round(recall, 4),
        "auc"      : round(auc, 4)
    }


def plot_confusion_matrix(y_true, y_pred,
                          model_name: str = "Model") -> None:
   
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Failure", "Failure"],
        yticklabels=["No Failure", "Failure"]
    )
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()

    os.makedirs(FIGURES_PATH, exist_ok=True)
    path = os.path.join(FIGURES_PATH, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] Confusion matrix saved → {path}")


def plot_roc_curve(y_true, y_prob,
                   model_name: str = "Model") -> None:
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc          = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr,
             color="steelblue",
             lw=2,
             label=f"ROC Curve (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1],
             color="gray",
             linestyle="--",
             label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()

    os.makedirs(FIGURES_PATH, exist_ok=True)
    path = os.path.join(FIGURES_PATH, "roc_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] ROC curve saved → {path}")


def save_results(metrics: dict,
                 path: str = "reports/results.json") -> None:

    import json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[SAVE] Results saved → {path}")


def cross_validate_scores(X, y, model, cv: int = 5) -> dict:
    
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.metrics import make_scorer

    print(f"\n[CV] Running {cv}-Fold Stratified Cross Validation...")

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    scoring = {
        "f1"  : make_scorer(f1_score),
        "auc" : make_scorer(roc_auc_score, response_method="predict_proba")
    }

    results = cross_validate(
        model, X, y,
        cv=skf,
        scoring=scoring,
        return_train_score=False
    )

    f1_mean  = results["test_f1"].mean()
    f1_std   = results["test_f1"].std()
    auc_mean = results["test_auc"].mean()
    auc_std  = results["test_auc"].std()

    print(f"[CV] F1  : {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"[CV] AUC : {auc_mean:.4f} ± {auc_std:.4f}")

    return {
        "cv_folds"   : cv,
        "f1_mean"    : round(f1_mean, 4),
        "f1_std"     : round(f1_std, 4),
        "auc_mean"   : round(auc_mean, 4),
        "auc_std"    : round(auc_std, 4)
    }


def save_full_results(metrics: dict,
                      cv_scores: dict,
                      path: str = "reports/results.json") -> None:

    import json

    full = {
        "model"    : metrics.get("model", "Unknown"),
        "metrics"  : metrics,
        "cv_scores": cv_scores
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(full, f, indent=4)
    print(f"[SAVE] Full results saved → {path}")


if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression

    np.random.seed(42)
    y_true = np.array([0]*100 + [1]*10)
    y_prob = np.random.rand(110)
    y_pred = (y_prob > 0.5).astype(int)


    metrics = evaluate_model(y_true, y_pred, y_prob,
                             model_name="Test Run")

    plot_confusion_matrix(y_true, y_pred, model_name="Test Run")
    plot_roc_curve(y_true, y_prob, model_name="Test Run")

    X_dummy = np.random.rand(110, 5)
    model   = LogisticRegression(random_state=42)
    cv_scores = cross_validate_scores(X_dummy, y_true, model, cv=5)


    save_full_results(metrics, cv_scores)

    print("\n[INFO] evaluate.py — all functions verified!")