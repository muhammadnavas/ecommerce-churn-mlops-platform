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
    
    rejected_rows = []
    clean_rows = []
    
    for idx, row in df.iterrows():
        reasons = []
        
        # Check 1: Missing Customer ID
        cid = str(row.get("CustomerID", "")).strip()
        if not cid or cid == "nan" or cid == "None" or cid == "<NA>":
            reasons.append("MISSING_CUSTOMER_ID")
            
        # Check 2: Unit Price validation
        try:
            price = float(row.get("UnitPrice", 0))
            if price <= 0:
                reasons.append("INVALID_UNIT_PRICE")
        except (ValueError, TypeError):
            reasons.append("NON_NUMERIC_PRICE")
            price = 0.0
            
        # Check 3: Quantity validation
        try:
            qty = int(row.get("Quantity", 0))
            if qty == 0:
                reasons.append("ZERO_QUANTITY")
        except (ValueError, TypeError):
            reasons.append("NON_INTEGER_QUANTITY")
            qty = 0
            
        # Check 4: Date parsing validation
        date_str = str(row.get("InvoiceDate", "")).strip()
        parsed_date = None
        try:
            parsed_date = pd.to_datetime(date_str, format="%Y-%m-%d %H:%M:%S", errors="raise")
        except Exception:
            reasons.append("CORRUPT_DATE_FORMAT")
            
        # Check 5: Cancellation consistency
        inv_no = str(row.get("InvoiceNo", "")).strip()
        is_cancelled = inv_no.startswith("C")
        if not is_cancelled and qty < 0:
            reasons.append("NEGATIVE_QUANTITY_UNSOLICITED")
            
        if reasons:
            rejected_dict = row.to_dict()
            rejected_dict["rejection_reason"] = "|".join(reasons)
            rejected_dict["rejected_at"] = datetime.now().isoformat()
            rejected_rows.append(rejected_dict)
        else:
            clean_dict = row.to_dict()
            clean_dict["CustomerID"] = str(cid).split(".")[0] # Clean integer format
            clean_dict["InvoiceDate"] = parsed_date.strftime("%Y-%m-%d %H:%M:%S")
            clean_dict["UnitPrice"] = price
            clean_dict["Quantity"] = qty
            clean_dict["is_cancelled"] = is_cancelled
            clean_dict["TotalAmount"] = round(qty * price, 2)
            clean_dict["Country"] = str(row.get("Country", config["etl"]["default_country"])).strip()
            clean_rows.append(clean_dict)
            
    # Save rejected records
    if rejected_rows:
        df_rejected = pd.DataFrame(rejected_rows)
        df_rejected.to_csv(rejected_log_path, index=False)
        print(f"[Data Quality] Isolated {len(df_rejected)} invalid records to {rejected_log_path}.")
    else:
        df_rejected = pd.DataFrame()
        
    # Save cleaned transactions
    df_clean = pd.DataFrame(clean_rows)
    clean_output_path = os.path.join(staging_dir, "cleaned_transactions.csv")
    df_clean.to_csv(clean_output_path, index=False)
    
    print(f"[Cleaning] Saved {len(df_clean)} cleaned records to {clean_output_path}.")
    return df_clean, df_rejected

if __name__ == "__main__":
    run_cleaning()
