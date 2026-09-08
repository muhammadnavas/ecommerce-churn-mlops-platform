"""
Unit tests for ETL Ingestion, Cleaning, and Star Schema Data Warehouse.
"""

import os
import sqlite3
import pandas as pd
import pytest

from src.etl.ingest import run_ingestion
from src.etl.clean import run_cleaning
from src.etl.transform import run_transform

def test_ingestion_and_cleaning():
    df_clean, df_rejected = run_cleaning()
    assert len(df_clean) > 0, "Cleaned dataset should not be empty."
    assert len(df_rejected) > 0, "Rejected records log should contain rejected records."
    assert "MISSING_CUSTOMER_ID" in df_rejected["rejection_reason"].to_string()

def test_star_schema_tables():
    rfm_df = run_transform()
    assert len(rfm_df) > 0, "RFM data mart should be populated."
    
    db_path = "data/warehouse/ecommerce_dw.db"
    assert os.path.exists(db_path), "Data warehouse DB should exist."
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT count(*) FROM fact_sales;")
    fact_count = cursor.fetchone()[0]
    cursor.execute("SELECT count(*) FROM dim_customer;")
    cust_count = cursor.fetchone()[0]
    conn.close()
    
    assert fact_count > 0, "fact_sales should contain records."
    assert cust_count > 0, "dim_customer should contain records."
