"""
Data Drift & Model Monitoring Module.
Calculates Kolmogorov-Smirnov (KS) test and Population Stability Index (PSI)
to monitor feature distribution changes between baseline and new production data.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import yaml

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def calculate_psi(reference, current, num_bins=10):
    """Calculates Population Stability Index (PSI) between reference and current feature data."""
    ref_clean = np.array(reference.dropna())
    curr_clean = np.array(current.dropna())
    
    if len(ref_clean) == 0 or len(curr_clean) == 0:
        return 0.0
        
    percentiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(ref_clean, percentiles)
    bins[0] = -np.inf
    bins[-1] = np.inf
    
    ref_counts, _ = np.histogram(ref_clean, bins=bins)
    curr_counts, _ = np.histogram(curr_clean, bins=bins)
    
    ref_pct = ref_counts / len(ref_clean)
    curr_pct = curr_counts / len(curr_clean)
    
    # Handle zero probabilities
    ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
    curr_pct = np.where(curr_pct == 0, 0.0001, curr_pct)
    
    psi_value = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
    return round(float(psi_value), 4)

def check_feature_drift(reference_df, current_df, feature_cols):
    drift_report = {}
    total_drifted = 0
    
    for col in feature_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
            
        ref_vals = reference_df[col]
        curr_vals = current_df[col]
        
        # KS-test
        ks_stat, p_value = ks_2samp(ref_vals, curr_vals)
        psi_val = calculate_psi(ref_vals, curr_vals)
        
        is_drifted = p_value < 0.05 or psi_val > 0.1
        if is_drifted:
            total_drifted += 1
            
        drift_report[col] = {
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_value), 4),
            "psi": psi_val,
            "drift_detected": bool(is_drifted)
        }
        
    overall_status = "NO_DRIFT"
    if total_drifted >= 3:
        overall_status = "HIGH_DRIFT"
    elif total_drifted > 0:
        overall_status = "MODERATE_DRIFT"
        
    return {
        "status": overall_status,
        "drifted_features_count": total_drifted,
        "total_features": len(feature_cols),
        "feature_details": drift_report,
        "checked_at": pd.Timestamp.now().isoformat()
    }

if __name__ == "__main__":
    from src.ml.feature_engineering import extract_features
    X, y, cust, cols = extract_features()
    # Test drift checking with simulated split
    ref = X.sample(frac=0.5, random_state=42)
    curr = X.drop(ref.index)
    report = check_feature_drift(ref, curr, cols)
    print("Drift Monitoring Report:", report["status"], f"({report['drifted_features_count']} drifted features)")
