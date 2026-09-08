"""
Script to extract and split the real-world UCI Online Retail Excel dataset
(c:\\Users\\Navas\\Downloads\\online+retail\\Online Retail.xlsx)
into monthly batch CSV files inside data/raw/ to replace synthetic mock data.
"""

import os
import pandas as pd

def convert_excel_to_monthly_batches(
    excel_path=r"c:\Users\Navas\Downloads\online+retail\Online Retail.xlsx",
    output_dir="data/raw"
):
    print(f"[Excel Converter] Reading dataset from {excel_path}...")
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel file not found at {excel_path}")
        
    df = pd.read_excel(excel_path)
    print(f"[Excel Converter] Loaded {len(df)} total records.")
    
    # Ensure InvoiceDate is datetime
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    
    # Extract Year and Month for batch naming: batch_YYYY_MM.csv
    df["year_month"] = df["InvoiceDate"].dt.strftime("%Y_%m")
    
    # Ensure raw data directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean up old synthetic batch files if any
    for existing_file in os.listdir(output_dir):
        if existing_file.startswith("batch_") and existing_file.endswith(".csv"):
            os.remove(os.path.join(output_dir, existing_file))
            
    # Save monthly batches
    grouped = df.groupby("year_month")
    batch_count = 0
    total_exported = 0
    
    for ym, group in grouped:
        # Drop the temporary grouping column
        batch_df = group.drop(columns=["year_month"]).copy()
        
        # Format InvoiceDate as string ISO format YYYY-MM-DD HH:MM:SS
        batch_df["InvoiceDate"] = batch_df["InvoiceDate"].dt.strftime("%Y-%m-%d %H:%M:%S")
        
        batch_filename = os.path.join(output_dir, f"batch_{ym}.csv")
        batch_df.to_csv(batch_filename, index=False)
        batch_count += 1
        total_exported += len(batch_df)
        print(f"[Excel Converter] Wrote {len(batch_df):>6} rows to {batch_filename}")
        
    print(f"[Excel Converter] Completed: Extracted {batch_count} monthly batches ({total_exported} rows) into {output_dir}.")

if __name__ == "__main__":
    convert_excel_to_monthly_batches()
