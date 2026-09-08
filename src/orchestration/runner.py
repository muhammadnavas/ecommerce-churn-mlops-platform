"""
Reproducible Pipeline Orchestrator for E-Commerce Data Engineering & MLOps Platform.
Executes complete sequence: Data Ingestion -> Cleaning/Validation -> Star Schema Transformation -> MLOps Model Training.
"""

import os
import sys
import time
from datetime import datetime

# Add project root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.etl.generate_sample_data import generate_ecommerce_dataset
from src.etl.ingest import run_ingestion
from src.etl.clean import run_cleaning
from src.etl.transform import run_transform
from src.ml.train import train_and_evaluate

def execute_pipeline():
    start_time = time.time()
    print("=" * 70)
    print(f"STARTING E-COMMERCE DATA PIPELINE & MLOPS WORKFLOW [{datetime.now().isoformat()}]")
    print("=" * 70)
    
    # Step 1: Ensure Raw Dataset
    raw_dir = "data/raw"
    if not os.path.exists(raw_dir) or len(os.listdir(raw_dir)) == 0:
        print("\n[Step 1/5] Generating initial batch datasets...")
        generate_ecommerce_dataset(output_dir=raw_dir)
    else:
        print("\n[Step 1/5] Existing raw batch datasets verified.")
        
    # Step 2: Data Ingestion
    print("\n[Step 2/5] Running Data Ingestion...")
    run_ingestion()
    
    # Step 3: Data Quality Validation & Cleaning
    print("\n[Step 3/5] Running Data Quality Checks & Cleaning...")
    run_cleaning()
    
    # Step 4: Star Schema & RFM Data Mart Transformation
    print("\n[Step 4/5] Building Star Schema & Customer RFM Mart...")
    run_transform()
    
    # Step 5: MLOps Model Training & Registry Export
    print("\n[Step 5/5] Training Models & Tracking Experiments in MLflow...")
    train_and_evaluate()
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"PIPELINE EXECUTION COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS!")
    print("=" * 70)

if __name__ == "__main__":
    execute_pipeline()
