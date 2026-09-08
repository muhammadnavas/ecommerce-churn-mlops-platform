"""
Feature Engineering module for Customer Churn Prediction.
Extracts customer behavioral, RFM, and transaction features from Star Schema DB.
"""

import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def extract_features(db_path="data/warehouse/ecommerce_dw.db"):
    conn = sqlite3.connect(db_path)
    
    # Extract customer RFM mart
    rfm_df = pd.read_sql("SELECT * FROM fact_customer_rfm", conn)
    
    # Extract additional product diversity & activity month counts from fact_sales
    sales_df = pd.read_sql("""
        SELECT 
            customer_id,
            COUNT(DISTINCT stock_code) as distinct_products,
            COUNT(DISTINCT substr(invoice_date, 1, 7)) as active_months
        FROM fact_sales
        WHERE is_cancelled = 0
        GROUP BY customer_id
    """, conn)
    
    conn.close()
    
    df_merged = pd.merge(rfm_df, sales_df, on="customer_id", how="left").fillna(0)
    
    feature_cols = [
        "recency_days",
        "frequency",
        "monetary",
        "avg_order_value",
        "total_items",
        "cancellation_rate",
        "distinct_products",
        "active_months"
    ]
    
    X = df_merged[feature_cols]
    y = df_merged["churn_label"]
    customer_ids = df_merged["customer_id"]
    
    return X, y, customer_ids, feature_cols

def prepare_train_test_split(config_path="config/config.yaml"):
    config = load_config(config_path)
    db_path = config["paths"]["warehouse_db_path"]
    test_size = config["mlops"]["test_size"]
    random_state = config["mlops"]["random_state"]
    
    X, y, customer_ids, feature_cols = extract_features(db_path)
    
    X_train, X_test, y_train, y_test, cust_train, cust_test = train_test_split(
        X, y, customer_ids, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"[Feature Engineering] Dataset split into Train ({len(X_train)} samples) and Test ({len(X_test)} samples).")
    print(f"[Feature Engineering] Churn balance - Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%}.")
    
    return X_train, X_test, y_train, y_test, feature_cols

if __name__ == "__main__":
    prepare_train_test_split()
