"""
FastAPI Inference Service for Customer Churn Prediction & Model Monitoring.
"""

import os
import json
import sqlite3
import joblib
import pandas as pd
import numpy as np
from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, status
import yaml

from src.ml.drift_monitor import check_feature_drift

app = FastAPI(
    title="E-Commerce Customer Churn Prediction API",
    description="Production REST API for real-time customer churn scoring, batch inference, and model drift monitoring.",
    version="1.0.0"
)

# Global variables for model, scaler, and metadata
MODEL = None
SCALER = None
METADATA = None
CONFIG = None

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

@app.on_event("startup")
def load_artifacts():
    global MODEL, SCALER, METADATA, CONFIG
    CONFIG = load_config()
    models_dir = CONFIG["paths"]["models_dir"]
    
    model_path = os.path.join(models_dir, "best_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    meta_path = os.path.join(models_dir, "model_metadata.json")
    
    if os.path.exists(model_path):
        MODEL = joblib.load(model_path)
    if os.path.exists(scaler_path):
        SCALER = joblib.load(scaler_path)
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            METADATA = json.load(f)

class CustomerFeatures(BaseModel):
    customer_id: Optional[str] = Field("12005", description="Customer unique ID")
    recency_days: float = Field(..., description="Days since last purchase")
    frequency: float = Field(..., description="Total number of completed orders")
    monetary: float = Field(..., description="Total expenditure")
    avg_order_value: float = Field(..., description="Average revenue per order")
    total_items: float = Field(..., description="Total product items purchased")
    cancellation_rate: float = Field(0.0, description="Ratio of cancelled orders")
    distinct_products: float = Field(..., description="Number of distinct products bought")
    active_months: float = Field(..., description="Number of active purchase months")

class PredictionOutput(BaseModel):
    customer_id: str
    churn_prediction: int
    churn_probability: float
    risk_level: str
    recommendation: str

class BatchPredictionRequest(BaseModel):
    customers: List[CustomerFeatures]

@app.get("/")
def read_root():
    return {
        "service": "E-Commerce Churn Prediction API",
        "status": "ONLINE",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
def check_health():
    db_path = CONFIG["paths"]["warehouse_db_path"] if CONFIG else "data/warehouse/ecommerce_dw.db"
    db_connected = os.path.exists(db_path)
    
    return {
        "status": "HEALTHY" if (MODEL and db_connected) else "DEGRADED",
        "model_loaded": MODEL is not None,
        "best_model_name": METADATA.get("best_model_name") if METADATA else "None",
        "best_roc_auc": METADATA.get("best_roc_auc") if METADATA else None,
        "database_connected": db_connected
    }

def format_risk(prob: float):
    if prob >= 0.70:
        return "HIGH_RISK", "Critical churn risk. Offer immediate re-engagement discount or dedicated support."
    elif prob >= 0.40:
        return "MEDIUM_RISK", "Moderate churn risk. Include in monthly promotional campaign."
    else:
        return "LOW_RISK", "Active loyal customer. Recommend upsell and VIP loyalty rewards."

@app.post("/predict", response_model=PredictionOutput)
def predict_churn(customer: CustomerFeatures):
    if not MODEL:
        raise HTTPException(status_code=500, detail="Model is not loaded. Please train models first.")
        
    feat_cols = METADATA["feature_cols"]
    input_df = pd.DataFrame([{
        "recency_days": customer.recency_days,
        "frequency": customer.frequency,
        "monetary": customer.monetary,
        "avg_order_value": customer.avg_order_value,
        "total_items": customer.total_items,
        "cancellation_rate": customer.cancellation_rate,
        "distinct_products": customer.distinct_products,
        "active_months": customer.active_months
    }])[feat_cols]
    
    if METADATA["best_model_name"] == "logistic_regression" and SCALER:
        X_eval = SCALER.transform(input_df)
    else:
        X_eval = input_df
        
    pred = int(MODEL.predict(X_eval)[0])
    prob = float(MODEL.predict_proba(X_eval)[0, 1])
    risk, rec = format_risk(prob)
    
    return PredictionOutput(
        customer_id=customer.customer_id or "UNKNOWN",
        churn_prediction=pred,
        churn_probability=round(prob, 4),
        risk_level=risk,
        recommendation=rec
    )

@app.post("/predict/batch")
def predict_batch(request: BatchPredictionRequest):
    results = [predict_churn(c) for c in request.customers]
    return {
        "total_processed": len(results),
        "predictions": results
    }

@app.get("/drift-status")
def get_drift_status():
    db_path = CONFIG["paths"]["warehouse_db_path"] if CONFIG else "data/warehouse/ecommerce_dw.db"
    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail="Warehouse DB not found.")
        
    conn = sqlite3.connect(db_path)
    df_rfm = pd.read_sql("SELECT * FROM fact_customer_rfm", conn)
    sales = pd.read_sql("""
        SELECT customer_id, COUNT(DISTINCT stock_code) as distinct_products,
               COUNT(DISTINCT substr(invoice_date, 1, 7)) as active_months
        FROM fact_sales WHERE is_cancelled = 0 GROUP BY customer_id
    """, conn)
    conn.close()
    
    df_all = pd.merge(df_rfm, sales, on="customer_id", how="left").fillna(0)
    feat_cols = METADATA["feature_cols"] if METADATA else [
        "recency_days", "frequency", "monetary", "avg_order_value",
        "total_items", "cancellation_rate", "distinct_products", "active_months"
    ]
    
    ref = df_all.sample(frac=0.5, random_state=42)
    curr = df_all.drop(ref.index)
    
    report = check_feature_drift(ref, curr, feat_cols)
    return report

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
