# 🚀 Predictive Maintenance using IoT Edge AI

A machine learning project that analyzes IoT sensor data to predict potential machine failures before they occur. The project includes data preprocessing, exploratory data analysis, feature engineering, model training, evaluation, explainability, and an interactive dashboard.

---

## 👥 Team

| Member      | Role                          | Branch                |
| ----------- | ----------------------------- | --------------------- |
| Swayam Arya | ML Engineer + GitHub Manager  | feature/swayam-ml     |
| Vrushabh    | Data Engineer                 | feature/vrushabh-data |
| Keshav      | Data Analysis & Visualization | feature/keshava-viz   |

---

## 🧠 Project Overview
 
This project uses the AI4I 2020 Predictive Maintenance dataset to build a predictive maintenance workflow. It combines data analysis, machine learning, and model explainability to identify conditions associated with machine failure.

The repository currently contains:

- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Machine learning using LightGBM
- Class imbalance handling using SMOTE
- Model evaluation
- SHAP-based explainability
- Streamlit dashboard
- Batch inference pipeline.

---

## 🎯 Problem Statement

Unexpected machine failures can lead to production downtime and increased maintenance costs.

The objective of this project is to predict machine failures using sensor measurements so that maintenance can be scheduled before failures occur.
  
---


## 🛠 Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | LightGBM |
| Imbalanced Learning | SMOTE |
| Explainability | SHAP |
| Visualization | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit |

---

## 📊 Dataset

**Dataset:** AI4I 2020 Predictive Maintenance Dataset

Main features include:

- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear
- Machine failure label

---

## 🏗 Workflow

```text
Sensor Data
      │
      ▼
Data Preprocessing
      │
      ▼
Feature Engineering
      │
      ▼
Model Training
      │
      ▼
Model Evaluation
      │
      ▼
SHAP Explainability
      │
      ▼
Inference
      │
      ▼
Dashboard
```

---

## 📁 Project Structure

```text
predictive-maintenance/
│
├── dashboard/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── reports/
├── src/
├── tests/
├── requirements.txt
└── README.md
```

---

## ✨ Features

- Data preprocessing
- Exploratory data analysis
- Feature engineering
- LightGBM classification model
- Class imbalance handling with SMOTE
- Model evaluation
- SHAP explainability
- Batch inference pipeline
- Streamlit dashboard

---

## 🤖 Methodology

1. Load and preprocess data.
2. Engineer additional features.
3. Balance the training data using SMOTE.
4. Train a LightGBM classifier.
5. Evaluate model performance.
6. Interpret predictions using SHAP.
7. Run inference on new samples.

---

## 📈 Model Performance

| Metric                   | Value     |
| ------------------------ | --------- |
| Macro F1 (CV with SMOTE) | 0.8819    |
| ROC-AUC                  | 0.9644    |
| Tuned Macro F1           | 0.8753    |
| Optimal Threshold        | 0.3988    |
| Noise Robustness         | up to 20% |

---

## 🚨 Alert System

| Level  | Probability | Action                |
| ------ | ----------- | --------------------- |
| GREEN  | < 0.30      | Normal operation      |
| YELLOW | 0.30–0.60   | Monitor closely       |
| RED    | ≥ 0.60      | Immediate maintenance |

---

## 📊 Visualizations

The repository includes visualizations for:

- Exploratory data analysis
- Correlation analysis
- Confusion matrix
- ROC curve
- Precision–Recall curve
- SHAP feature importance

---

## ⚙ Installation

Clone the repository:

```bash
git clone <repository-url>
cd predictive-maintenance
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶ Usage

Run the notebooks in order:

```text
01_eda.ipynb
02_features.ipynb
03_model.ipynb
04_noise_threshold_analysis.ipynb
05_pipeline_documentation.ipynb
```

Launch the dashboard:

```bash
streamlit run dashboard/app.py
```

---
