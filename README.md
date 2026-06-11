# Predictive Maintenance - IoT Edge AI

## Project Overview
AI-powered system to predict machine failures before 
they happen using IoT sensor data and external 
environmental signals.

## Team
- Swayam Arya - ML Engineer & GitHub Manager
- Vrushabh - Data Engineer  
- Keshav - Analysis & Visualization

## Tech Stack
- Python, LightGBM, SMOTE
- Pandas, NumPy, SHAP
- Scikit-learn, Matplotlib

## Week-wise Progress
- [x] Week 1: Data ingestion & EDA ✅
  - Dataset loaded (10,000 rows, 14 columns)
  - Rolling mean, std, variance calculated
  - Class imbalance: 96.6% non-failure
  - LightGBM basic model trained
  - Feature importance: Tool_wear_min most important
- [ ] Week 2: Feature engineering
- [ ] Week 3: ML modelingss
- [ ] Week 4: Noise analysis & deployment



## Data Pipeline (Week 1 — Vrushabh)
|--------------------------------------------------------|
|      File         |    Shape    |      Description     |
|-------------------|-------------|----------------------|
| clean_data.csv    | (10000, 7)  | Cleaned raw data     |
| featured_data.csv | (10000, 19) | 8 new features added |
| X_train.csv       | (7000, 18)  | Training features    |
| X_val.csv         | (1500, 18)  | Validation features  |
| X_test.csv        | (1500, 18)  | Test features        |
| y_train.csv       | (7000, 1)   | Training labels      |
| y_val.csv         | (1500, 1)   | Validation labels    |
| y_test.csv        | (1500, 1)   | Test labels          |
|--------------------------------------------------------|

## ML WORK (Week 1 - Swayam)

|-------------------------------------------------------------------------------------|
| Task                           |   Status    |           Description                |
|-------------------------------------------------------------------------------------|
| Dataset Analysis               | Completed   | AI4I dataset explored and validated  |
| Time-Series Processing         | Completed   | Sensor logs processed                |
| Rolling Mean                   | Completed   | Operational window mean calculated   |
| Rolling Standard Deviation     | Completed   | Signal variability measured          |
| Signal Variance                | Completed   | Variance features generated          |
| Operational Window Features    | Completed   | Window-based statistics created      |
| Baseline Feature Engineering   | Completed   | Initial ML features prepared         |
| Class Imbalance Analysis       | Completed   | Failure distribution analyzed        |
| Feature Importance Analysis    | Completed   | Key predictive features identified   |
| SHAP Explainability            | Completed   | Model interpretation completed       |
| model.py (LightGBM Pipeline)   | Completed   | Training and evaluation module added |
|-------------------------------------------------------------------------------------|