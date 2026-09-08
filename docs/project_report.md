# Project Submission Report
## E-Commerce Sales Analytics & Customer Churn Prediction Platform

**Course**: Data Engineering and MLOps  
**Mode**: Individual Project  
**Date**: September 2026  

---

## 1. Executive Summary

This project presents an end-to-end Data Engineering & MLOps platform for processing high-volume online retail transactions, maintaining a dimensional Star Schema Data Warehouse, generating business intelligence visual analytics, and operationalizing machine learning models for customer churn prediction.

### Key Highlights & Results:
- **ETL Pipeline**: Ingested **9,486 raw records** across 12 monthly batch files (`batch_2010_12.csv` to `batch_2011_11.csv`).
- **Data Quality & Rejection**: Automatically isolated **144 corrupted/invalid records** (missing customer IDs, negative prices, zero quantities, bad date formats) into `logs/rejected_records.csv`.
- **Star Schema Data Warehouse**: Transformed **9,342 valid transaction records** into SQLite star schema tables (`dim_customer`, `dim_product`, `dim_geography`, `dim_date`, `fact_sales`) and built the analytical data mart `fact_customer_rfm` for 336 customers.
- **MLOps Model Benchmark**: Evaluated Logistic Regression, Random Forest, and XGBoost using a 90-day inactivity churn target.
  - **Random Forest**: ROC-AUC = **1.0000**, F1-Score = **1.0000**
  - **XGBoost**: ROC-AUC = **1.0000**, F1-Score = **1.0000**
  - **Logistic Regression**: ROC-AUC = **0.9982**, F1-Score = **0.9615**
- **Experiment Tracking**: Tracked all runs, hyper-parameters, and model artifacts with **MLflow**.
- **Serving & Dashboard**: Implemented a **FastAPI** REST inference engine (`/predict`, `/predict/batch`, `/drift-status`) and a **6-tab interactive Streamlit Dashboard** with real-time churn prediction calculator and Kolmogorov-Smirnov feature drift monitoring.

---

## 2. Part 1: Data Engineering & ETL Pipeline

### 2.1 Ingestion Architecture & Incremental Batch Extractor
The ingestion framework (`src/etl/ingest.py`) reads monthly transaction extracts from `data/raw/` in landing area fashion. Key metadata (extraction timestamp, file status, raw row counts) is recorded in `logs/ingestion_history.csv` to ensure auditability and lineage tracking.

```
data/raw/batch_*.csv -> Landing Zone -> data/staging/raw_combined.csv
```

### 2.2 Data Quality Rules & Rejection System
To ensure data integrity, `src/etl/clean.py` executes 5 rigorous validation rules before allowing records into staging:
1. `MISSING_CUSTOMER_ID`: Rejects records where CustomerID is NULL or empty.
2. `INVALID_UNIT_PRICE`: Rejects UnitPrice <= 0.
3. `ZERO_QUANTITY`: Rejects Quantity == 0.
4. `CORRUPT_DATE_FORMAT`: Rejects unparseable InvoiceDate strings.
5. `NEGATIVE_QUANTITY_UNSOLICITED`: Rejects negative quantities on non-cancellation orders.

All rejected records are written to `logs/rejected_records.csv` with reason codes.

### 2.3 Dimensional Data Warehouse Schema
The transformed dataset is stored in SQLite (`data/warehouse/ecommerce_dw.db`):

- **`dim_customer`**: 336 customer records with primary country, first/last purchase timestamps, and total completed order counts.
- **`dim_product`**: 30 distinct products with item descriptions and average prices.
- **`dim_geography`**: 10 countries (United Kingdom, France, Germany, EIRE, Spain, Netherlands, Belgium, etc.).
- **`dim_date`**: Calendar lookup table with year, quarter, month, day, day_of_week, and weekend indicators.
- **`fact_sales`**: 9,342 transaction fact records containing quantity, unit price, line revenue total, and cancellation status.
- **`fact_customer_rfm`**: Analytical data mart summarizing Recency, Frequency, Monetary value, Average Order Value, Total Items, Cancellation Rate, and Churn Label.

---

## 3. Part 2: MLOps Extension & Model Pipeline

### 3.1 Churn Definition & Feature Engineering
Customer churn is defined using a **90-day inactivity cutoff**:
$$\text{Churn Label} = \begin{cases} 1 & \text{if } \text{Recency Days} > 90 \\ 0 & \text{otherwise} \end{cases}$$

Features extracted per customer (`src/ml/feature_engineering.py`):
- `recency_days`: Days elapsed since last order.
- `frequency`: Total number of completed orders.
- `monetary`: Total cumulative spend (£).
- `avg_order_value`: Average spend per order.
- `total_items`: Total volume of items purchased.
- `cancellation_rate`: Ratio of cancelled invoices.
- `distinct_products`: Product catalog breadth.
- `active_months`: Count of distinct purchase months.

### 3.2 Model Training & MLflow Experimentation
Three classification models were trained and tracked using MLflow (`src/ml/train.py`):

| Model Name | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Random Forest (100 Trees, Max Depth 6)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **XGBoost (100 Trees, LR 0.1)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Logistic Regression** | **0.9706** | **1.0000** | **0.9259** | **0.9615** | **0.9982** |

The **Random Forest** model was selected as the production model and exported to `models/best_model.pkl`.

### 3.3 Data Drift & Model Monitoring
`src/ml/drift_monitor.py` implements statistical monitoring using two-sample **Kolmogorov-Smirnov (KS) tests** and **Population Stability Index (PSI)**:
$$\text{PSI} = \sum \left( P_{\text{current}} - P_{\text{reference}} \right) \times \ln\left( \frac{P_{\text{current}}}{P_{\text{reference}}} \right)$$
Features with PSI > 0.1 or KS p-value < 0.05 are flagged for potential retraining.

---

## 4. Serving API & Interactive Dashboard

### 4.1 FastAPI Inference Service (`src/api/main.py`)
- `GET /health`: Health status & loaded model version.
- `POST /predict`: Real-time single customer churn risk prediction, probability score, risk tier (`HIGH_RISK`, `MEDIUM_RISK`, `LOW_RISK`), and business recommendations.
- `POST /predict/batch`: High-performance batch customer risk evaluation.
- `GET /drift-status`: Returns feature drift assessment.

### 4.2 Streamlit Analytical Dashboard (`src/dashboard/app.py`)
Features 6 interactive tabs:
1. **Executive Overview**: Total Revenue, Active Customers, Order Volume trends over time.
2. **Product Analytics**: Top 10 revenue-generating products & price distribution histograms.
3. **Customer RFM**: Recency vs Spend scatter plots & customer segment share (VIP, Loyal, At-Risk).
4. **Geo Distribution**: Country-wise revenue breakdowns and ranking table.
5. **Retention & Cohorts**: Portfolio churn rate & active vs churned customer profile comparison.
6. **ML Churn Predictor**: Real-time slider risk calculator, customer ID lookup, model benchmarking, and drift monitoring.

---

## 5. Execution & Verification Evidence

### Verification Summary
- **Pipeline Runner Execution**: `python src/orchestration/runner.py` completes all 5 ETL & MLOps steps in ~56 seconds.
- **Unit Test Suite**: `python -m pytest tests/` passes 4/4 test cases cleanly.
- **Containerization**: `docker-compose -f docker/docker-compose.yml up` builds and runs API and Dashboard containers seamlessly.

---

## 6. Conclusion & Recommendations

The platform successfully satisfies all Part 1 and Part 2 requirements of Project 3:
1. **Reliable Retail Data Pipeline**: Robust incremental batch ingestion, data validation error logging, and Star Schema Data Warehousing.
2. **Business Intelligence**: Rich 6-tab interactive Streamlit analytical application.
3. **End-to-End MLOps Solution**: Experimentation tracking with MLflow, registered top-performing Random Forest model, FastAPI REST inference engine, and statistical drift monitoring.
