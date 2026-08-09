# 💰 AI Budget Deviation Prediction System

An end-to-end Enterprise Artificial Intelligence System designed to predict corporate budget deviations, forecast actual expenses, categorize financial risk levels, analyze root-cause cost drivers, and recommend actionable cost optimization strategies.

---

## 🏗️ System Architecture

```
[1. Synthetic Data Generator]  --> 15,000 Records x 21 Financial Schema Attributes
              |
              v
[2. Schema & Quality Validator] --> Multi-rule Schema & Boundary Verification
              |
              v
[3. Data Preprocessing Pipeline]--> Median Imputation, IQR Capping, One-Hot Encoding, StandardScaler
              |
              +----------------------------+
              |                            |
              v                            v
[4. Exploratory Data Analysis]  [5. Feature Engineering Engine]
    - Univariate & Bivariate        - Budget Utilization Ratios
    - Departmental Dynamics          - Operational Efficiency Metrics
    - Financial Trend Reports        - Inflation & Temporal Dynamics
                                           |
                                           v
                                [6. Model Training & Cross-Validation]
                                    - Regressors: HistGradientBoosting, RandomForest, Ridge
                                    - Classifiers: Risk Tier Categorization
                                    - Automated Metric Evaluation (RMSE, MAE, R², F1)
                                           |
                                           v
                                [7. Production Inference & Explainability]
                                    - Global Feature Importance
                                    - Local Cost Driver Decomposition
                                    - Targeted Mitigation Recommendations
                                           |
                                           v
                                [8. Streamlit Web Dashboard]
                                    - Executive KPI Cards
                                    - Interactive Plotly Analytics
                                    - Real-Time AI Prediction Tool
```

---

## 📁 Repository Structure

```
AI-Budget-Deviation-Prediction/
├── data/
│   ├── raw/
│   │   └── financial_budget_deviation_raw.csv
│   └── processed/
│       ├── financial_budget_deviation_clean.csv
│       ├── financial_budget_deviation_clean.parquet
│       ├── financial_budget_deviation_ml.csv
│       └── financial_budget_deviation_ml.parquet
├── models/
│   ├── encoder.joblib
│   ├── scaler.joblib
│   ├── preprocessing_pipeline.joblib
│   ├── budget_deviation_regressor.joblib
│   └── budget_deviation_classifier.joblib
├── reports/
│   ├── eda/
│   ├── preprocessing_report.json
│   ├── validation_report.json
│   └── model_evaluation_report.json
├── src/
│   ├── config.py
│   ├── data_generator.py
│   ├── generate_dataset.py
│   ├── validator.py
│   ├── eda/
│   │   ├── univariate_analysis.py
│   │   ├── bivariate_analysis.py
│   │   ├── multivariate_analysis.py
│   │   ├── department_analysis.py
│   │   ├── financial_analysis.py
│   │   ├── trend_analysis.py
│   │   ├── insight_generator.py
│   │   └── report_generator.py
│   ├── preprocessing/
│   │   ├── schema_validator.py
│   │   ├── missing_value_handler.py
│   │   ├── outlier_handler.py
│   │   ├── encoder.py
│   │   ├── scaler.py
│   │   └── preprocessing_pipeline.py
│   ├── features/
│   │   ├── feature_builder.py
│   │   └── feature_selector.py
│   ├── models/
│   │   ├── model_trainer.py
│   │   └── model_evaluator.py
│   ├── explainability/
│   │   └── explainability.py
│   └── inference/
│       └── predictor.py
├── streamlit_app/
│   └── app.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_inference.py
├── run_eda.py
├── run_training.py
├── requirements.txt
└── README.md
```

---

## 🚀 Execution & Usage Guide

### 1. Environment Setup
Activate virtual environment and install dependencies:
```bash
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Generate Synthetic Dataset
Generate 15,000 realistic financial records with departmental rules, seasonal trends, missing values, and anomalies:
```bash
.\venv\Scripts\python.exe src/generate_dataset.py
```

### 3. Run Preprocessing Pipeline
Execute schema validation, missing value imputation, IQR outlier capping, categorical encoding, scaling, and artifact saving:
```bash
.\venv\Scripts\python.exe src/preprocessing/preprocess_dataset.py
```

### 4. Run Exploratory Data Analysis (EDA)
Generate statistical profilers, departmental breakdowns, financial trend plots, and comprehensive HTML report:
```bash
.\venv\Scripts\python.exe run_eda.py
```

### 5. Train Machine Learning Models
Train candidate regressors and classifiers with 5-fold cross-validation, evaluate benchmarks, and save models:
```bash
.\venv\Scripts\python.exe run_training.py
```

### 6. Launch Streamlit Interactive Web Application
Launch the interactive web dashboard:
```bash
.\venv\Scripts\streamlit.exe run streamlit_app/app.py
```

### 7. Run Automated Pytest Suite
Run unit tests to verify system integrity:
```bash
.\venv\Scripts\python.exe -m pytest tests/
```

---

## 📊 Model Performance & Benchmarks

- **Continuous Budget Deviation Regressor**: Best model selected via 5-Fold Cross-Validation. Evaluated using RMSE, MAE, R², and MAPE metrics.
- **Risk Categorization Classifier**: Classifies financial records into 4 risk tiers (`Underbudget`, `Low Risk`, `Moderate Overbudget`, `Severe Overbudget`) evaluated using Precision, Recall, and F1-Score.

---

## 🛡️ License & Engineering Standards

- Developed following Clean Architecture, SOLID principles, type annotations, and docstrings.
- PEP8 compliant codebase.
