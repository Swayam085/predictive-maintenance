# 🔧 Predictive Maintenance — IoT Edge AI
> Internship Project | Infotact Solutions & Co.

## 👥 Team
| Member | Role | Branch |
|---|---|---|
| Swayam Arya | ML Engineer + GitHub Manager | feature/swayam-ml |
| Vrushabh | Data Engineer | feature/vrushabh-data |
| Keshav | Analysis + Visualization | feature/keshava-viz |

## 🛠️ Tech Stack
- **Language:** Python 3.x
- **ML Model:** LightGBM
- **Balancing:** SMOTE
- **Explainability:** SHAP
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Data:** Pandas, NumPy

## 📊 Dataset
- **Name:** AI4I 2020 Predictive Maintenance
- **Records:** 10,000
- **Features:** Air Temp, Process Temp, Rotational Speed, Torque, Tool Wear, Machine Failure

## 📈 Key Findings (EDA + Model)
- 3.3% failure rate — severe class imbalance
- **Tool Wear** is #1 failure predictor (SHAP confirmed)
- High Torque + Low Rotational Speed = high risk zone
- Air & Process temperature highly correlated (0.87)
- Model evaluated with Confusion Matrix, ROC-AUC, Precision-Recall

## 📊 Visualization Highlights (23 Figures)
1–9: EDA & Feature Analysis  
10–12: Rolling Statistics  
13–15: SHAP Explainability  
16–18: Class Imbalance, Risk Zones, Tool Wear  
19–23: Model Evaluation (Confusion Matrix, ROC, PR Curve, Predictions, Feature Importance)

👉 Full report: [`reports/final_eda_report.html`](reports/final_eda_report.html)

## 🚀 How to Run
```bash
pip install -r requirements.txt

jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/keshava_viz_day4.ipynb
jupyter notebook notebooks/03_model.ipynb
```
=
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


## ML WORK (Week 1 - Swayam)

| Task | Status | Description |
|------|--------|-------------|
| Dataset Analysis | ✅ | AI4I dataset explored and validated |
| Time-Series Processing | ✅ | Sensor logs processed |
| Rolling Mean | ✅ | Operational window mean calculated |
| Rolling Std Deviation | ✅ | Signal variability measured |
| Signal Variance | ✅ | Variance features generated |
| Operational Window | ✅ | Window-based statistics created |
| Baseline Feature Engineering | ✅ | Initial ML features prepared |
| Class Imbalance Analysis | ✅ | Failure distribution analyzed |
| Feature Importance | ✅ | Key predictive features identified |
| SHAP Analysis | ✅ | Model interpretation completed |
| model.py (LightGBM) | ✅ | Training and evaluation module added |

