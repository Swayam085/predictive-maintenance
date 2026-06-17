# 🔧 Predictive Maintenance — IoT Edge AI
> Internship Project | Infotact Solutions & Co.

## 👥 Team
| Member | Role | Branch |
|---|---|---|
| Swayam Arya | ML Engineer + GitHub Manager | feature/swayam-ml |
| Vrushabh | Data Engineer | feature/vrushabh-data |
| Keshav | Analysis + Visualization | feature/keshava-viz |

## 📁 Project Structure
predictive-maintenance/

├── data/

│   ├── raw/                  ← ai4i2020.csv

│   └── processed/            ← clean, featured, train/test splits

├── notebooks/

│   ├── 01_eda.ipynb

│   ├── 02_features.ipynb

│   ├── 03_model.ipynb

│   └── keshava_viz_day4.ipynb ← SHAP + Model Evaluation Viz

├── src/

│   ├── data_loader.py

│   ├── features.py

│   ├── model.py

│   └── evaluate.py

├── reports/

│   ├── figures/               ← 23 visualization PNGs

│   ├── final_eda_report.html  ← Complete EDA Report

│   ├── interactice_toolwear_box.html

│   └── eda_summary.csv

└── requirements.txt

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