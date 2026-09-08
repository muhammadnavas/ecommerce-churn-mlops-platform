"""
Star Schema Transformation & Data Mart Builder.
Populates SQLite/PostgreSQL Data Warehouse with dimension tables,
fact_sales, and fact_customer_rfm analytical marts.
"""

import os
import sqlite3
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yaml

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_db_connection(db_path):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    return conn

def init_tables(conn):
    cursor = conn.cursor()
    
    # 1. dim_geography
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dim_geography (
            country_id INTEGER PRIMARY KEY AUTOINCREMENT,
            country_name TEXT UNIQUE NOT NULL
        );
    """)
    
    # 2. dim_customer
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dim_customer (
            customer_id TEXT PRIMARY KEY,
            primary_country TEXT,
            first_purchase_date TEXT,
            last_purchase_date TEXT,
            total_orders INTEGER
        );
    """)
    
    # 3. dim_product
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dim_product (
            stock_code TEXT PRIMARY KEY,
            description TEXT,
            avg_unit_price REAL
        );
    """)
    
    # 4. dim_date
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dim_date (
            date_key INTEGER PRIMARY KEY,
            full_date TEXT UNIQUE NOT NULL,
            year INTEGER,
            quarter INTEGER,
            month INTEGER,
            day INTEGER,
            day_of_week INTEGER,
            is_weekend INTEGER
        );
    """)
    
    # 5. fact_sales
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fact_sales (
            sales_id INTEGER PRIMARY KEY AUTOINCREMENT,
            invoice_no TEXT NOT NULL,
            customer_id TEXT,
            stock_code TEXT,
            country_id INTEGER,
            date_key INTEGER,
            invoice_date TEXT,
            quantity INTEGER,
            unit_price REAL,
            total_amount REAL,
            is_cancelled INTEGER,
            FOREIGN KEY (customer_id) REFERENCES dim_customer(customer_id),
            FOREIGN KEY (stock_code) REFERENCES dim_product(stock_code),
            FOREIGN KEY (country_id) REFERENCES dim_geography(country_id),
            FOREIGN KEY (date_key) REFERENCES dim_date(date_key)
        );
    """)
    
    # 6. fact_customer_rfm (Customer Analytics Mart)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fact_customer_rfm (
            customer_id TEXT PRIMARY KEY,
            snapshot_date TEXT,
            recency_days INTEGER,
            frequency INTEGER,
            monetary REAL,
            avg_order_value REAL,
            total_items INTEGER,
            cancellation_rate REAL,
            churn_label INTEGER,
            FOREIGN KEY (customer_id) REFERENCES dim_customer(customer_id)
        );
    """)
    
    conn.commit()

def run_transform(cleaned_path="data/staging/cleaned_transactions.csv", config_path="config/config.yaml"):
    config = load_config(config_path)
    db_path = config["paths"]["warehouse_db_path"]
    churn_days = config["mlops"]["churn_inactivity_days"]
    
    if not os.path.exists(cleaned_path):
        raise FileNotFoundError(f"Cleaned dataset {cleaned_path} not found. Please run cleaning first.")
        
    df = pd.read_csv(cleaned_path, dtype={"CustomerID": "str", "StockCode": "str", "InvoiceNo": "str"})
    df["InvoiceDate_dt"] = pd.to_datetime(df["InvoiceDate"])
    
    conn = get_db_connection(db_path)
    init_tables(conn)
    
    # 1. Populate dim_geography
    unique_countries = df["Country"].dropna().unique()
    geo_df = pd.DataFrame({"country_name": unique_countries})
    geo_df.to_sql("dim_geography", conn, if_exists="replace", index_label="country_id")
    # Query lookup map
    geo_map = pd.read_sql("SELECT country_id, country_name FROM dim_geography", conn)
    country_to_id = dict(zip(geo_map["country_name"], geo_map["country_id"]))
    
    # 2. Populate dim_product
    product_grouped = df.groupby("StockCode").agg(
        description=("Description", "first"),
        avg_unit_price=("UnitPrice", "mean")
    ).reset_index()
    product_grouped.rename(columns={"StockCode": "stock_code"}, inplace=True)
    product_grouped.to_sql("dim_product", conn, if_exists="replace", index=False)
    
    # 3. Populate dim_date
    df["full_date"] = df["InvoiceDate_dt"].dt.strftime("%Y-%m-%d")
    dates = pd.to_datetime(df["full_date"].unique())
    dim_date = pd.DataFrame({
        "full_date": dates.strftime("%Y-%m-%d"),
        "date_key": dates.strftime("%Y%m%d").astype(int),
        "year": dates.year,
        "quarter": dates.quarter,
        "month": dates.month,
        "day": dates.day,
        "day_of_week": dates.dayofweek,
        "is_weekend": (dates.dayofweek >= 5).astype(int)
    })
    dim_date.to_sql("dim_date", conn, if_exists="replace", index=False)
    date_to_key = dict(zip(dim_date["full_date"], dim_date["date_key"]))
    
    # 4. Populate dim_customer
    non_cancel_df = df[df["is_cancelled"] == False]
    cust_agg = df.groupby("CustomerID").agg(
        primary_country=("Country", "first"),
        first_purchase_date=("InvoiceDate", "min"),
        last_purchase_date=("InvoiceDate", "max"),
        total_orders=("InvoiceNo", "nunique")
    ).reset_index()
    cust_agg.rename(columns={"CustomerID": "customer_id"}, inplace=True)
    cust_agg.to_sql("dim_customer", conn, if_exists="replace", index=False)
    
    # 5. Populate fact_sales
    df["country_id"] = df["Country"].map(country_to_id)
    df["date_key"] = df["full_date"].map(date_to_key)
    
    fact_sales_df = pd.DataFrame({
        "invoice_no": df["InvoiceNo"],
        "customer_id": df["CustomerID"],
        "stock_code": df["StockCode"],
        "country_id": df["country_id"],
        "date_key": df["date_key"],
        "invoice_date": df["InvoiceDate"],
        "quantity": df["Quantity"],
        "unit_price": df["UnitPrice"],
        "total_amount": df["TotalAmount"],
        "is_cancelled": df["is_cancelled"].astype(int)
    })
    fact_sales_df.to_sql("fact_sales", conn, if_exists="replace", index_label="sales_id")
    
    # 6. Populate fact_customer_rfm (Analytics Data Mart & Churn Target)
    max_date = df["InvoiceDate_dt"].max()
    snapshot_date = max_date + timedelta(days=1)
    
    rfm_rows = []
    for cid, group in df.groupby("CustomerID"):
        valid_orders = group[group["is_cancelled"] == False]
        last_dt = group["InvoiceDate_dt"].max()
        recency = (snapshot_date - last_dt).days
        frequency = valid_orders["InvoiceNo"].nunique()
        monetary = round(valid_orders["TotalAmount"].sum(), 2)
        avg_ov = round(monetary / frequency, 2) if frequency > 0 else 0.0
        total_items = valid_orders["Quantity"].sum()
        total_attempts = group["InvoiceNo"].nunique()
        cancels = group[group["is_cancelled"] == True]["InvoiceNo"].nunique()
        cancel_rate = round(cancels / total_attempts, 4) if total_attempts > 0 else 0.0
        
        # Churn defined as inactivity > churn_inactivity_days (e.g. 90 days)
        churn_label = 1 if recency > churn_days else 0
        
        rfm_rows.append({
            "customer_id": cid,
            "snapshot_date": snapshot_date.strftime("%Y-%m-%d"),
            "recency_days": recency,
            "frequency": frequency,
            "monetary": monetary,
            "avg_order_value": avg_ov,
            "total_items": total_items,
            "cancellation_rate": cancel_rate,
            "churn_label": churn_label
        })
        
    rfm_df = pd.DataFrame(rfm_rows)
    rfm_df.to_sql("fact_customer_rfm", conn, if_exists="replace", index=False)
    
    conn.commit()
    conn.close()
    
    print(f"[Transform] Star Schema populated in SQLite DB {db_path}.")
    print(f"[Transform] fact_sales rows: {len(fact_sales_df)}, fact_customer_rfm rows: {len(rfm_df)}.")
    return rfm_df

if __name__ == "__main__":
    run_transform()
