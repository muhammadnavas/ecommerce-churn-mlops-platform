# E-Commerce Sales Analytics & Customer Churn Prediction Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.22%2B-FF4B4B.svg)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-2.0%2B-0194E2.svg)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

Production-grade Data Engineering and MLOps platform built for online retail transaction processing, dimensional data warehousing, interactive executive visual analytics, and automated customer churn prediction.

---

## 🌟 Key Features

- **Automated Data Ingestion & Quality Control**:
  - Incremental batch ingestion of multi-month transaction extracts (`data/raw/batch_*.csv`).
  - Automated rejection logging (`logs/rejected_records.csv`) for corrupt records (missing CustomerID, non-numeric prices, zero quantities, bad date formats).
- **Star Schema Data Warehouse**:
  - SQLite & PostgreSQL compatible dimensional model (`dim_customer`, `dim_product`, `dim_geography`, `dim_date`, `fact_sales`).
  - RFM Analytics Data Mart (`fact_customer_rfm`) with 90-day inactivity churn ground truth labels.
- **MLOps Experimentation & Model Pipeline**:
  - Features: Recency, Frequency, Monetary, Avg Order Value, Item Volume, Cancel Rate, Product Diversity, Active Months.
  - Models: **Logistic Regression**, **Random Forest**, and **XGBoost**.
  - Tracked in **MLflow** with model registry and export of best model (`models/best_model.pkl`).
  - **Drift Monitoring**: Kolmogorov-Smirnov (KS-test) and Population Stability Index (PSI) feature drift detection.
- **FastAPI Inference REST Service**:
  - `/health`, `/predict`, `/predict/batch`, and `/drift-status` endpoints.
- **6-Tab Interactive Streamlit Dashboard**:
  1. 📊 Executive Overview & Sales Performance
  2. 📦 Product Analytics & Category Distribution
  3. 👥 Customer RFM Segmentation Scatter Plot & Segments
  4. 🌍 Geographical Sales Distribution
  5. 🔄 Retention & Cohort Profile Comparison
  6. 🤖 Real-Time Churn Predictor Calculator & Model Monitoring

---

## 📂 Project Directory Structure

```
d:\Projects\Python\ML Project\
├── config/
│   └── config.yaml             # Central system configuration
├── dags/
│   └── ecommerce_etl_dag.py    # Apache Airflow DAG definition
├── data/
│   ├── raw/                    # Simulated monthly batch CSV extracts
│   ├── staging/                # Ingested & cleaned datasets
│   └── warehouse/              # SQLite Data Warehouse (ecommerce_dw.db & mlflow.db)
├── docs/
│   ├── architecture_diagram.md # Architecture & pipeline sequence diagram
│   ├── data_dictionary.md      # Schema DDL, field definitions, validation rules
│   └── project_report.md       # Comprehensive Part 1 & Part 2 submission report
├── docker/
│   ├── Dockerfile.api          # FastAPI container spec
│   ├── Dockerfile.dashboard    # Streamlit container spec
│   └── docker-compose.yml      # Container orchestration
├── logs/
│   ├── rejected_records.csv    # Data quality error logs
│   └── ingestion_history.csv   # Extraction metadata log
├── models/
│   ├── best_model.pkl          # Exported top-performing model artifact
│   ├── scaler.pkl              # Feature scaler
│   └── model_metadata.json     # Benchmark metrics & model metadata
├── src/
│   ├── etl/
│   │   ├── generate_sample_data.py # Realistic transaction data generator
│   │   ├── ingest.py           # Landing area raw batch ingestor
│   │   ├── clean.py            # Data quality validation & cleaner
│   │   └── transform.py        # Star schema & RFM data mart builder
│   ├── ml/
│   │   ├── feature_engineering.py # RFM feature extractor & train-test split
│   │   ├── train.py            # Model training & MLflow tracker
│   │   └── drift_monitor.py    # Statistical feature drift detector (KS & PSI)
│   ├── api/
│   │   └── main.py             # FastAPI prediction service
│   ├── dashboard/
│   │   └── app.py              # Streamlit 6-tab dashboard
│   └── orchestration/
│       └── runner.py           # Reproducible end-to-end pipeline runner
├── tests/
│   ├── test_etl.py             # ETL unit tests
│   └── test_api.py             # FastAPI unit tests
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Environment Setup
```bash
# Navigate to project directory
cd "d:\Projects\Python\ML Project"

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Complete End-to-End Pipeline
Executes batch ingestion -> data cleaning -> star schema transform -> feature extraction -> model training -> MLflow logging in one command:
```bash
python src/orchestration/runner.py
```

### 3. Launch Interactive Streamlit Dashboard
```bash
streamlit run src/dashboard/app.py
```
Open your browser to `http://localhost:8501`.

### 4. Launch FastAPI Prediction Service
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger documentation available at `http://localhost:8000/docs`.

### 5. Launch MLflow Experiment Tracking UI
```bash
mlflow ui --backend-store-uri sqlite:///data/warehouse/mlflow.db
```
Open `http://localhost:5000` to compare experiments, metrics, ROC curves, and parameters.

### 6. Run Unit Test Suite
```bash
python -m pytest tests/
```

### 7. Run with Docker Compose
```bash
docker-compose -f docker/docker-compose.yml up --build
```

---

## 📊 Analytical & MLOps Results

- **Data Processed**: 9,486 raw records -> 9,342 cleaned records -> 336 customer RFM profiles.
- **Data Quality**: 144 corrupt records isolated into `logs/rejected_records.csv`.
- **Best Model**: **Random Forest** (ROC-AUC = **1.0000**, F1-Score = **1.0000**).
- **Documentation**: Detailed reports available in `docs/project_report.md` and `docs/data_dictionary.md`.
