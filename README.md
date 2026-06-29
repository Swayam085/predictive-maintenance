# 🚀 Predictive Maintenance using IoT Edge AI

> Internship Project | Infotact Solutions & Co.

---

## 👥 Team

| Member      | Role                          | Branch                |
| ----------- | ----------------------------- | --------------------- |
| Swayam Arya | ML Engineer + GitHub Manager  | feature/swayam-ml     |
| Vrushabh    | Data Engineer                 | feature/vrushabh-data |
| Keshav      | Data Analysis & Visualization | feature/keshava-viz   |

---

## 🧠 Project Overview

This project focuses on building a **predictive maintenance system** using IoT sensor data to identify potential machine failures before they occur.

The solution integrates:

* Data preprocessing & feature engineering
* Exploratory Data Analysis (EDA)
* Machine Learning (LightGBM)
* Model explainability using SHAP

---

## 🛠️ Tech Stack

* **Language:** Python 3.x
* **ML Model:** LightGBM
* **Imbalance Handling:** SMOTE
* **Explainability:** SHAP
* **Visualization:** Matplotlib, Seaborn, Plotly
* **Data Processing:** Pandas, NumPy

---

## 📊 Dataset

* **Name:** AI4I 2020 Predictive Maintenance Dataset
* **Records:** 10,000
* **Features:**
  Air Temperature, Process Temperature, Rotational Speed, Torque, Tool Wear, Machine Failure

---

## 🔍 Key Insights

* ⚠️ Only **3.3% failures** → Severe class imbalance
* 🔧 **Tool Wear** is the most critical predictor (SHAP validated)
* ⚡ High Torque + Low Speed → High-risk operating zone
* 🌡️ Strong correlation (0.87) between Air & Process Temperature

---

## 📈 Model Performance

* Evaluated using:

  * Confusion Matrix
  * ROC-AUC Curve
  * Precision-Recall Curve
* Balanced dataset using **SMOTE**
* Interpreted predictions using **SHAP values**

---

## 📊 Visualization Highlights (23 Figures)

* EDA & Feature Analysis
* Rolling Statistics
* SHAP Explainability
* Risk Zones & Failure Patterns
* Model Evaluation Metrics

👉 Full Report: `reports/final_eda_report.html`

---

## 📂 Project Structure

```
├── data/
├── notebooks/
├── reports/
├── src/
├── figures/
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

```bash
pip install -r requirements.txt

jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/keshava_viz_day4.ipynb
jupyter notebook notebooks/03_model.ipynb
```

---

## 📅 Project Progress

* [x] Week 1: Data ingestion & EDA ✅
* [x] Week 2: Feature Engineering ✅
* [x] Week 3: Model Building & Evaluation ✅
* [ ] Week 4: Explainability & Final Report 

---

## 📊 Data Pipeline

| File              | Shape       | Description         |
| ----------------- | ----------- | ------------------- |
| clean_data.csv    | (10000, 7)  | Cleaned dataset     |
| featured_data.csv | (10000, 19) | Engineered features |
| X_train.csv       | (7000, 18)  | Training data       |
| X_val.csv         | (1500, 18)  | Validation data     |
| X_test.csv        | (1500, 18)  | Test data           |


## Evaluation Module (Week 2 — Vrushabh)

### Functions (9 total):
|         Function              |            Description         |         Output            |
|-------------------------------|--------------------------------|---------------------------|
| evaluate_model()              | F1, Precision, Recall, ROC-AUC | dict                      |
| plot_confusion_matrix()       | Heatmap plot                   | confusion_matrix.png      |
| plot_roc_curve()              | ROC curve plot                 | roc_curve.png             |
| plot_precision_recall_curve() | PR curve, optimal threshold    | pr_curve.png              |
| save_classification_report()  | Full report text               | classification_report.txt |
| cross_validate_scores()       | 5-fold stratified CV           | dict                      |
| save_full_results()           | Metrics + CV JSON              | results.json              |
| prepare_shap_background()     | SHAP background samples        | shap_background.csv       |
| verify_no_leakage()           | Index overlap check            | bool                      |
  
### Verified:
- No data leakage: 0 overlap rows
- SHAP background: (100, 19) ready for Swayam
- Optimal threshold: 0.2288 for LightGBM calibration
- Code review: docstrings + PEP8 + error handling applied


## Inference Pipeline (Week 3 — Vrushabh)

### Files added:
|         File        |                    Description                           |
|---------------------|----------------------------------------------------------|
| src/inference.py    | Raw sensor input → feature engineering → ONNX prediction |
| src/alert_system.py | GREEN / YELLOW / RED alert system with logging           |
| src/pipeline.py     | End-to-end batch pipeline with CSV support               |
| tests/              | 27 pytest unit tests — all passing                       |

### Alert Thresholds:
|  Level | Probability |        Action         |
|--------|-------------|-----------------------|
| GREEN  | < 0.30      | Normal operation      |
| YELLOW | 0.30 – 0.60 | Monitor closely       |
| RED    | >= 0.60     | Immediate maintenance |

### Verified:
- Preprocessing: raw dict → (1, 18) float32 array
- Batch pipeline: 10 CSV rows, 90% accuracy
- Alert logging: inference.log + alert_logs.json
- Unit tests: 27/27 PASS in 27.11s

### Week-wise Progress update:
- [x] Week 3: Inference + Alert pipeline complete

---

## 🎯 Impact

* Reduced unexpected failures using predictive insights
* Improved interpretability using SHAP
* Built a complete end-to-end ML pipeline

---

## 📌 Future Work

* Real-time IoT integration
* Edge deployment optimization
* Model monitoring dashboard

---

## ⭐ Conclusion

A complete **industry-level predictive maintenance pipeline** combining data engineering, machine learning, and explainability.


## Week 2 – Contextual Data Fusion (swayam-ml)

- Integrated contextual features with IoT sensor dataset
- Performed feature fusion and comparative analysis
- Conducted 5-Fold Cross Validation on fused dataset
- Baseline CV F1: 0.8195
- External Features CV F1: 0.8715
- Fused Dataset CV F1: 0.8696
- Cross Validation Std Dev: 0.0140
- Achieved project target (Macro F1 > 0.85)

See `notebooks/03_model.ipynb` for detailed analysis.


## Week 3 — Imbalanced Classification & LightGBM Modeling

### ML Track (Swayam Arya)

#### Key Results
| Metric | Value |
|--------|-------|
| Macro F1 (CV with SMOTE) | 0.8819 |
| Tuned Threshold | 0.3988 |
| Tuned Macro F1 | 0.8753 |
| ROC-AUC | 0.9644 |
| Noise Robust upto | 0.10 |
| Top Feature (SHAP) | Torque_Nm |

#### Work Done
- SMOTE implemented inside CV folds — no data leakage
- LightGBM hyperparameters tuned — n_estimators 500, num_leaves 63
- Threshold tuned from 0.50 to 0.3988 via PR curve
- Noise sensitivity analysis — model robust at all noise levels
- SHAP summary, bar and waterfall plots generated
- Confusion matrix and ROC-AUC updated with tuned model


## Week 4 — Explainability & Dashboard Integration

### ML Track (Swayam Arya)

#### Completed (Day 1 – Day 3)

| Module                              | Status |
| ----------------------------------- | ------ |
| Noise Sensitivity Analysis          | ✅      |
| SHAP Summary, Bar & Waterfall Plots | ✅      |
| Precision–Recall Curve              | ✅      |
| Streamlit Dashboard Integration     | ✅      |
| Dashboard Merge Conflict Resolution | ✅      |
| Figure Overlap Fix                  | ✅      |
| Dashboard Testing                   | ✅      |

#### Dashboard Features

* Live IoT sensor simulation
* SHAP Explainability tab
* Precision–Recall Curve visualization
* Noise Analysis dashboard
* Final Model Metrics dashboard
* Interactive machine health monitoring

#### Current Performance

| Metric           | Value           |
| ---------------- | --------------- |
| Macro F1 (CV)    | **0.8819**      |
| Tuned Macro F1   | **0.8753**      |
| ROC-AUC          | **0.9644**      |
| Tuned Threshold  | **0.3988**      |
| Noise Robustness | **20% Noise**   |
| Dashboard Status | ✅ Fully Working |
