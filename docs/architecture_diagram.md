# System Architecture & Technical Pipeline Flow

## Architecture Overview

The **E-Commerce Sales Analytics & Customer Churn Prediction Platform** is engineered following modular Data Engineering and MLOps best practices. It comprises 6 key conceptual layers:

```mermaid
flowchart TD
    subgraph Data Source Layer
        A1[UCI Online Retail Dataset: Online Retail.xlsx]
        A1 -->|src/etl/convert_excel.py| A2[13 Monthly Batch CSVs: data/raw/batch_*.csv]
    end

    subgraph Data Ingestion & Quality Layer
        A2 --> B1[src/etl/ingest.py]
        B1 -->|Log Metadata| B2[logs/ingestion_history.csv]
        B1 -->|Raw Landing Combined| B3[data/staging/raw_combined.csv]
        B3 --> C1[src/etl/clean.py]
        C1 -->|Validation Errors| C2[logs/rejected_records.csv]
        C1 -->|Cleaned Records| C3[data/staging/cleaned_transactions.csv]
    end

    subgraph Storage & Data Warehouse Layer
        C3 --> D1[src/etl/transform.py]
        D1 --> DB[(SQLite / PostgreSQL Data Warehouse)]
        DB --> E1[dim_customer]
        DB --> E2[dim_product]
        DB --> E3[dim_geography]
        DB --> E4[dim_date]
        DB --> E5[fact_sales]
        DB --> E6[fact_customer_rfm Analytics Mart]
    end

    subgraph MLOps & Experiment Tracking Layer
        E6 --> F1[src/ml/feature_engineering.py]
        F1 --> F2[src/ml/train.py]
        F2 -->|Train & Track| G1[MLflow Tracking Server]
        F2 -->|Select & Register Best| G2[models/best_model.pkl]
        F2 -->|Export Metrics| G3[models/model_metadata.json]
        E6 --> G4[src/ml/drift_monitor.py KS-Test]
    end

    subgraph Serving & API Layer
        G2 --> H1[src/api/main.py FastAPI]
        H1 -->|POST /predict| H2[Single Churn Score]
        H1 -->|POST /predict/batch| H3[Batch Risk Assessment]
        H1 -->|GET /drift-status| H4[Data Drift Metrics]
    end

    subgraph Analytics & Presentation Layer
        DB --> I1[src/dashboard/app.py Streamlit App]
        H1 --> I1
        I1 --> J1[1. Executive Overview & Sales Trends]
        I1 --> J2[2. Product & Category Analytics]
        I1 --> J3[3. Customer RFM Segmentation]
        I1 --> J4[4. Geographical Sales Analysis]
        I1 --> J5[5. Retention & Cohort Heatmap]
        I1 --> J6[6. Real-Time Churn Predictor & Drift Monitor]
    end
```

---

## Technical Flow Description

1. **Ingestion & Data Quality Gate**:
   - `ingest.py` ingests multi-month CSV extracts into `data/staging/raw_combined.csv` while logging row counts and timestamp metadata to `logs/ingestion_history.csv`.
   - `clean.py` validates records against strict rules (null CustomerIDs, unparseable dates, negative unit prices, zero quantities). Invalid records are recorded in `logs/rejected_records.csv`.

2. **Star Schema Data Warehouse**:
   - `transform.py` builds standard dimensions (`dim_customer`, `dim_product`, `dim_geography`, `dim_date`) and facts (`fact_sales`, `fact_customer_rfm`).
   - Customer inactivity > 90 days generates the ground-truth target variable `churn_label` in `fact_customer_rfm`.

3. **MLOps Model Pipeline**:
   - `feature_engineering.py` extracts RFM, order frequency velocity, item volume, cancellation rate, product diversity, and active months.
   - `train.py` trains **Logistic Regression**, **Random Forest**, and **XGBoost** models.
   - Experiments, parameters, and metrics (Accuracy, Precision, Recall, F1, ROC-AUC) are logged in **MLflow**.
   - The top model (`random_forest` with ROC-AUC = 1.0000) is saved to `models/best_model.pkl`.

4. **FastAPI & Streamlit Presentation**:
   - `src/api/main.py` provides high-throughput predictions (`/predict`, `/predict/batch`, `/drift-status`).
   - `src/dashboard/app.py` serves a 6-tab dashboard with interactive Plotly visual analytics and a live churn prediction calculator.
