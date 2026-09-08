"""
Ingestion module for loading raw batch CSV files, logging extraction metadata,
and preparing landing data.
"""

import os
import glob
from datetime import datetime
import pandas as pd
import yaml

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_ingestion(config_path="config/config.yaml"):
    config = load_config(config_path)
    raw_dir = config["paths"]["raw_data_dir"]
    staging_dir = config["paths"]["staging_data_dir"]
    os.makedirs(staging_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    batch_pattern = os.path.join(raw_dir, config["etl"]["batch_file_pattern"])
    batch_files = sorted(glob.glob(batch_pattern))
    
    if not batch_files:
        raise FileNotFoundError(f"No batch CSV files found matching {batch_pattern}")
        
    ingestion_summary = []
    all_raw_dfs = []
    
    for b_file in batch_files:
        filename = os.path.basename(b_file)
        try:
            df = pd.read_csv(b_file, dtype={"CustomerID": "str", "StockCode": "str", "InvoiceNo": "str"})
            df["source_file"] = filename
            df["ingested_at"] = datetime.now().isoformat()
            
            all_raw_dfs.append(df)
            ingestion_summary.append({
                "source_file": filename,
                "status": "SUCCESS",
                "extracted_rows": len(df),
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            ingestion_summary.append({
                "source_file": filename,
                "status": f"FAILED: {str(e)}",
                "extracted_rows": 0,
                "timestamp": datetime.now().isoformat()
            })
            
    combined_raw_df = pd.concat(all_raw_dfs, ignore_index=True)
    raw_output_path = os.path.join(staging_dir, "raw_combined.csv")
    combined_raw_df.to_csv(raw_output_path, index=False)
    
    log_df = pd.DataFrame(ingestion_summary)
    ingestion_log_path = "logs/ingestion_history.csv"
    log_df.to_csv(ingestion_log_path, index=False)
    
    print(f"[Ingestion] Successfully ingested {len(batch_files)} batches into {raw_output_path} ({len(combined_raw_df)} total raw rows).")
    return combined_raw_df, log_df

if __name__ == "__main__":
    run_ingestion()
