"""
Data Quality & Cleaning module.
Validates records, logs rejected records with reason codes,
standardizes currency/dates, handles cancellations, and saves cleaned data.
"""

import os
from datetime import datetime
import pandas as pd
import yaml

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_cleaning(input_path="data/staging/raw_combined.csv", config_path="config/config.yaml"):
    config = load_config(config_path)
    staging_dir = config["paths"]["staging_data_dir"]
    rejected_log_path = config["paths"]["rejected_records_log"]
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Staging file {input_path} not found. Please run ingestion first.")
        
    df = pd.read_csv(input_path, dtype={"CustomerID": "str", "StockCode": "str", "InvoiceNo": "str"})
    print(f"[Cleaning] Loaded {len(df)} records from staging...")

    # Vectorized validation masks
    # Check 1: Missing Customer ID
    cid_str = df["CustomerID"].astype(str).str.strip()
    missing_cid_mask = df["CustomerID"].isna() | cid_str.isin(["", "nan", "None", "<NA>", "NaN"])
    
    # Check 2: Unit Price validation
    numeric_price = pd.to_numeric(df["UnitPrice"], errors="coerce")
    invalid_price_mask = numeric_price.isna() | (numeric_price <= 0)
    
    # Check 3: Quantity validation
    numeric_qty = pd.to_numeric(df["Quantity"], errors="coerce")
    invalid_qty_mask = numeric_qty.isna() | (numeric_qty == 0)
    
    # Check 4: Date parsing validation
    parsed_dates = pd.to_datetime(df["InvoiceDate"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    corrupt_date_mask = parsed_dates.isna()
    
    # Check 5: Cancellation consistency
    inv_no_str = df["InvoiceNo"].astype(str).str.strip()
    is_cancelled_mask = inv_no_str.str.startswith("C")
    unsolicited_neg_qty_mask = (~is_cancelled_mask) & (numeric_qty < 0)

    # Combine masks to identify valid vs rejected
    has_error_mask = (
        missing_cid_mask |
        invalid_price_mask |
        invalid_qty_mask |
        corrupt_date_mask |
        unsolicited_neg_qty_mask
    )

    # Build rejection reason strings
    reasons = pd.Series("", index=df.index)
    reasons = reasons.where(~missing_cid_mask, reasons + "|MISSING_CUSTOMER_ID")
    reasons = reasons.where(~invalid_price_mask, reasons + "|INVALID_UNIT_PRICE")
    reasons = reasons.where(~invalid_qty_mask, reasons + "|ZERO_QUANTITY")
    reasons = reasons.where(~corrupt_date_mask, reasons + "|CORRUPT_DATE_FORMAT")
    reasons = reasons.where(~unsolicited_neg_qty_mask, reasons + "|NEGATIVE_QUANTITY_UNSOLICITED")
    reasons = reasons.str.lstrip("|")

    # Reject records
    df_rejected = df[has_error_mask].copy()
    if not df_rejected.empty:
        df_rejected["rejection_reason"] = reasons[has_error_mask]
        df_rejected["rejected_at"] = datetime.now().isoformat()
        os.makedirs(os.path.dirname(rejected_log_path), exist_ok=True)
        df_rejected.to_csv(rejected_log_path, index=False)
        print(f"[Data Quality] Isolated {len(df_rejected)} invalid records to {rejected_log_path}.")
    else:
        df_rejected = pd.DataFrame()

    # Clean valid records
    df_clean = df[~has_error_mask].copy()
    df_clean["CustomerID"] = cid_str[~has_error_mask].str.split(".").str[0]
    df_clean["InvoiceDate"] = parsed_dates[~has_error_mask].dt.strftime("%Y-%m-%d %H:%M:%S")
    df_clean["UnitPrice"] = numeric_price[~has_error_mask]
    df_clean["Quantity"] = numeric_qty[~has_error_mask].astype(int)
    df_clean["is_cancelled"] = is_cancelled_mask[~has_error_mask]
    df_clean["TotalAmount"] = (df_clean["Quantity"] * df_clean["UnitPrice"]).round(2)
    
    default_country = config["etl"].get("default_country", "United Kingdom")
    df_clean["Country"] = df_clean["Country"].astype(str).str.strip().replace("", default_country)
    
    # Save cleaned transactions
    clean_output_path = os.path.join(staging_dir, "cleaned_transactions.csv")
    df_clean.to_csv(clean_output_path, index=False)
    
    print(f"[Cleaning] Saved {len(df_clean)} cleaned records to {clean_output_path}.")
    return df_clean, df_rejected

if __name__ == "__main__":
    run_cleaning()

