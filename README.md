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
* [x] Week 4: Explainability & Final Report ✅

---

## 📊 Data Pipeline

| File              | Shape       | Description         |
| ----------------- | ----------- | ------------------- |
| clean_data.csv    | (10000, 7)  | Cleaned dataset     |
| featured_data.csv | (10000, 19) | Engineered features |
| X_train.csv       | (7000, 18)  | Training data       |
| X_val.csv         | (1500, 18)  | Validation data     |
| X_test.csv        | (1500, 18)  | Test data           |

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
