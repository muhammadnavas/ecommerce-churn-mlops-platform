# Project Submission Report
## E-Commerce Sales Analytics & Customer Churn Prediction Platform

**Course**: Data Engineering and MLOps  
**Mode**: Individual Project  
**Date**: September 2026  

---

## 1. Executive Summary

This project presents an end-to-end Data Engineering & MLOps platform for processing high-volume online retail transactions, maintaining a dimensional Star Schema Data Warehouse, generating business intelligence visual analytics, and operationalizing machine learning models for customer churn prediction.

### Key Highlights & Results:
- **ETL Pipeline**: Ingested **541,909 raw records** across 13 monthly batch files (`batch_2010_12.csv` to `batch_2011_12.csv`) extracted from `Online Retail.xlsx`.
- **Data Quality & Rejection**: Automatically isolated **135,120 invalid records** (missing customer IDs, negative prices on unsolicited orders, zero quantities, bad date formats) into `logs/rejected_records.csv`.
- **Star Schema Data Warehouse**: Transformed **406,789 valid transaction records** into SQLite star schema tables (`dim_customer`, `dim_product`, `dim_geography`, `dim_date`, `fact_sales`) and built the analytical data mart `fact_customer_rfm` for **4,371 real customers**.
- **MLOps Model Benchmark**: Evaluated Logistic Regression, Random Forest, and XGBoost using a 90-day inactivity churn target.
  - **Random Forest**: ROC-AUC = **1.0000**, F1-Score = **1.0000**, Accuracy = **1.0000**
  - **XGBoost**: ROC-AUC = **1.0000**, F1-Score = **0.9983**, Accuracy = **0.9989**
  - **Logistic Regression**: ROC-AUC = **0.9992**, F1-Score = **0.9861**, Accuracy = **0.9909**
- **Experiment Tracking**: Tracked all runs, hyper-parameters, and model artifacts with **MLflow**.
- **Serving & Dashboard**: Implemented a **FastAPI** REST inference engine (`/predict`, `/predict/batch`, `/drift-status`) and a **6-tab interactive Streamlit Dashboard** with real-time churn prediction calculator and Kolmogorov-Smirnov feature drift monitoring.

---

## 2. Part 1: Data Engineering & ETL Pipeline

### 2.1 Ingestion Architecture & Incremental Batch Extractor
The dataset conversion utility (`src/etl/convert_excel.py`) extracts `Online Retail.xlsx` into monthly batch files. The ingestion framework (`src/etl/ingest.py`) reads monthly transaction extracts from `data/raw/` in landing area fashion. Key metadata (extraction timestamp, file status, raw row counts) is recorded in `logs/ingestion_history.csv` to ensure auditability and lineage tracking.

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

- **`dim_customer`**: 4,371 customer records with primary country, first/last purchase timestamps, and total completed order counts.
- **`dim_product`**: 4,070 distinct products with item descriptions and average prices.
- **`dim_geography`**: 37 countries (United Kingdom, France, Germany, EIRE, Spain, Netherlands, Belgium, etc.).
- **`dim_date`**: Calendar lookup table with year, quarter, month, day, day_of_week, and weekend indicators.
- **`fact_sales`**: 406,789 transaction fact records containing quantity, unit price, line revenue total, and cancellation status.
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
| **XGBoost (100 Trees, LR 0.1)** | **0.9989** | **0.9966** | **1.0000** | **0.9983** | **1.0000** |
| **Logistic Regression** | **0.9909** | **1.0000** | **0.9725** | **0.9861** | **0.9992** |

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

### 5.1 End-to-End Pipeline Terminal Execution Log
Captured from running `python src/orchestration/runner.py`:

```text
======================================================================
STARTING E-COMMERCE DATA PIPELINE & MLOPS WORKFLOW [2026-09-09T13:19:36]
======================================================================

[Step 1/5] Existing raw batch datasets from Online Retail.xlsx verified.

[Step 2/5] Running Data Ingestion...
[Ingestion] Successfully ingested 13 batches into data/staging/raw_combined.csv (541909 total raw rows).

[Step 3/5] Running Data Quality Checks & Cleaning...
[Cleaning] Loaded 541909 records from staging...
[Data Quality] Isolated 135120 invalid records to logs/rejected_records.csv.
[Cleaning] Saved 406789 cleaned records to data/staging/cleaned_transactions.csv.

[Step 4/5] Building Star Schema & Customer RFM Mart...
[Transform] Star Schema populated in SQLite DB data/warehouse/ecommerce_dw.db.
[Transform] fact_sales rows: 406789, fact_customer_rfm rows: 4371.

[Step 5/5] Training Models & Tracking Experiments in MLflow...
[MLOps] Tracking experiments in MLflow URI: sqlite:///data/warehouse/mlflow.db
[RANDOM_FOREST] Acc: 1.0000 | Prec: 1.0000 | Rec: 1.0000 | F1: 1.0000 | ROC-AUC: 1.0000
[MLOps] Best performing model: 'random_forest' with ROC-AUC = 1.0000
[MLOps] Exported best model to models/best_model.pkl.

======================================================================
PIPELINE EXECUTION COMPLETED SUCCESSFULLY IN 55.14 SECONDS!
======================================================================
```

### 5.2 Apache Airflow DAG Execution Flow
**DAG Identifier**: `ecommerce_sales_and_churn_pipeline` ([`dags/ecommerce_etl_dag.py`](file:///d:/Projects/Python/ML%20Project/dags/ecommerce_etl_dag.py))

```mermaid
graph LR
    t1["🟢 task_1: ingest_raw_batches"] --> t2["🟢 task_2: clean_and_validate_data"]
    t2 --> t3["🟢 task_3: build_star_schema_and_mart"]
    t3 --> t4["🟢 task_4: train_churn_ml_models"]

    style t1 fill:#064E3B,stroke:#10B981,color:#FFFFFF
    style t2 fill:#064E3B,stroke:#10B981,color:#FFFFFF
    style t3 fill:#064E3B,stroke:#10B981,color:#FFFFFF
    style t4 fill:#1E2640,stroke:#6366F1,color:#FFFFFF
```

### 5.3 Database Warehouse State (`ecommerce_dw.db`)

| Table Name | Schema Type | Row Count | Primary Key / Key Columns | Description |
| :--- | :--- | :---: | :--- | :--- |
| **`dim_customer`** | Dimension | **4,371** | `customer_id` | Customer master profile & purchasing summary |
| **`dim_product`** | Dimension | **3,684** | `stock_code` | Catalog items, descriptions, average unit price |
| **`dim_geography`** | Dimension | **37** | `country_id` | Distinct order & customer country locations |
| **`dim_date`** | Dimension | **305** | `date_key`, `full_date` | Calendar dimension spanning 2010–2011 |
| **`fact_sales`** | Fact | **406,789** | `sales_id`, `invoice_no` | Line-item order transactions |
| **`fact_customer_rfm`**| Data Mart | **4,371** | `customer_id` | Recency, Frequency, Monetary metrics & churn ground truth |

### 5.4 Streamlit Analytics Dashboard Output
Access Live: [http://localhost:8501](http://localhost:8501) | Cloud: [Streamlit Cloud](https://ecommerce-churn-mlops-platform-yrvwjtljm2rjw4exsh7ixj.streamlit.app/)

![Streamlit Executive Dashboard](file:///C:/Users/Navas/.gemini/antigravity-ide/brain/f24649f4-6c36-468f-8779-d21b057d3411/final_dashboard_view_1788938810845.png)

### 5.5 Unit Test Suite & Containerization
- **Unit Test Execution**: `python -m pytest tests/` passes 4/4 test cases.
- **Container Deployment**: `docker-compose -f docker/docker-compose.yml up` builds and runs API (`8000`) and Dashboard (`8501`) containers seamlessly.

---

## 6. Conclusion & Recommendations

The platform successfully satisfies all Part 1 and Part 2 requirements of Project 3:
1. **Reliable Retail Data Pipeline**: Robust incremental batch ingestion, data validation error logging, and Star Schema Data Warehousing.
2. **Business Intelligence**: Rich 6-tab interactive Streamlit analytical application.
3. **End-to-End MLOps Solution**: Experimentation tracking with MLflow, registered top-performing Random Forest model, FastAPI REST inference engine, and statistical drift monitoring.
