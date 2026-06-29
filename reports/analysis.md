# Predictive Maintenance – Analysis Report

## 1. Objective

The goal of this project is to analyze machine sensor data and predict potential failures using machine learning models. The analysis focuses on understanding feature behavior, detecting patterns, and evaluating model performance.

---

## 2. Data Understanding

The dataset contains operational and sensor-related features such as:

* Air Temperature
* Process Temperature
* Rotational Speed
* Torque
* Tool Wear
* Failure Type (Target Variable)

### Key Observations:

* The dataset includes both **numerical features** and **categorical failure labels**.
* There is **class imbalance**, with normal operations dominating failure cases.

---

## 3. Exploratory Data Analysis (EDA)

### 3.1 Class Distribution

* Majority of samples belong to **non-failure class**.
* Minority failure classes can affect model learning.

### 3.2 Feature Distributions

* Most features follow near-normal distributions.
* Tool wear shows skewness, indicating progressive degradation.

### 3.3 Correlation Analysis

* Strong relationships observed between:

  * Air Temperature & Process Temperature
* Weak correlation between:

  * Torque and rotational speed (likely nonlinear)

---

## 4. Feature Behavior Insights

### Temperature Features

* Higher temperature differences may indicate stress conditions.
* Sudden spikes correlate with certain failure types.

### Torque vs Speed

* Nonlinear relationship observed.
* High torque at low speed may indicate overload.

### Tool Wear

* Strongest indicator of failure.
* Increased wear significantly raises failure probability.

---

## 5. Rolling Statistics Analysis

### Rolling Mean & Standard Deviation

* Helps capture trends over time.
* Increasing standard deviation indicates instability.

### Signal Variance

* High variance corresponds to abnormal machine behavior.

---

## 6. Model Performance Analysis

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

### Confusion Matrix

* Model favors majority class.
* Some failure cases misclassified.

### ROC Curve

* Shows strong class separability.

### Precision-Recall Curve

* Better reflects performance under class imbalance.

---

## 7. Feature Importance

### Key Features:

* Tool Wear
* Torque
* Rotational Speed
* Temperature Difference

### Insights:

* Tool wear is the dominant predictor.
* Combined feature effects improve prediction accuracy.

---

## 8. Model Comparison

### Cross-Validation

* Stable performance across folds.

### Ablation Study

* Removing key features reduces accuracy.
* Confirms importance of engineered features.

---

## 9. Failure Pattern Insights

* Tool wear is the primary failure driver.
* High torque + high wear increases risk.
* Failures cluster in specific feature ranges.

---

## 10. Issues Identified

### Data Issues

* Class imbalance
* Sensor noise

### Model Issues

* Bias toward majority class
* Difficulty detecting rare failures

### Repository Issues

* File naming inconsistencies
* Merge conflicts due to deleted vs modified files

---

## 11. Recommendations

### Data

* Apply resampling techniques (SMOTE / undersampling)

### Model

* Use ensemble methods (LightGBM, XGBoost)
* Tune hyperparameters

### Visualization

* Maintain consistent file naming

### Git Workflow

* Use `git mv` for renaming
* Sync branches before changes

---

## 12. Conclusion

The project successfully identifies key factors influencing machine failure. Tool wear, torque, and temperature variations are the most critical indicators. The model performs well overall but can be improved by addressing class imbalance and refining feature engineering. The system demonstrates strong potential for real-world predictive maintenance applications.

---
