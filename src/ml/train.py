"""
Model Training & MLOps Experiment Tracking.
Trains Logistic Regression, Random Forest, and XGBoost models,
logs metrics & artifacts with MLflow (with fallback), and exports the best model.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import joblib
import pandas as pd
import numpy as np
import yaml

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# Optional MLflow import fallback
try:
    import mlflow
    import mlflow.sklearn
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

from src.ml.feature_engineering import prepare_train_test_split

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train_and_evaluate(config_path="config/config.yaml"):
    config = load_config(config_path)
    models_dir = config["paths"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)
    
    # Configure MLflow if available
    if HAS_MLFLOW:
        try:
            mlflow_uri = config["paths"]["mlflow_tracking_uri"]
            mlflow.set_tracking_uri(mlflow_uri)
            experiment_name = config["mlops"]["experiment_name"]
            mlflow.set_experiment(experiment_name)
            print(f"[MLOps] Tracking experiments in MLflow URI: {mlflow_uri}")
        except Exception as e:
            print(f"[MLOps Warning] Could not set MLflow URI: {e}")
    else:
        print("[MLOps Warning] MLflow module not found. Falling back to local artifact logging.")
        
    X_train, X_test, y_train, y_test, feature_cols = prepare_train_test_split(config_path)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        "xgboost": XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, eval_metric="logloss")
    }
    
    best_score = -1.0
    best_model_name = None
    best_model_obj = None
    results = {}
    
    for name, model in models.items():
        if HAS_MLFLOW:
            try:
                mlflow.start_run(run_name=name)
            except Exception:
                pass
                
        # Fit model
        if name == "logistic_regression":
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            probs = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]
            
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        roc_auc = roc_auc_score(y_test, probs)
        
        metrics_dict = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc)
        }
        
        if HAS_MLFLOW:
            try:
                mlflow.log_param("model_type", name)
                mlflow.log_metrics(metrics_dict)
                mlflow.sklearn.log_model(model, artifact_path=f"model_{name}")
                mlflow.end_run()
            except Exception as e:
                print(f"[MLOps Warning] Failed to log model {name} to MLflow: {e}")
                
        results[name] = metrics_dict
        print(f"[{name.upper()}] Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")
        
        if roc_auc > best_score:
            best_score = roc_auc
            best_model_name = name
            best_model_obj = model
            
    print(f"\n[MLOps] Best performing model: '{best_model_name}' with ROC-AUC = {best_score:.4f}")
    
    # Save best model and scaler locally for fast inference serving
    model_save_path = os.path.join(models_dir, "best_model.pkl")
    scaler_save_path = os.path.join(models_dir, "scaler.pkl")
    metadata_path = os.path.join(models_dir, "model_metadata.json")
    
    joblib.dump(best_model_obj, model_save_path)
    joblib.dump(scaler, scaler_save_path)
    
    metadata = {
        "best_model_name": best_model_name,
        "best_roc_auc": float(best_score),
        "feature_cols": feature_cols,
        "results": results,
        "trained_at": pd.Timestamp.now().isoformat()
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"[MLOps] Exported best model to {model_save_path} and metadata to {metadata_path}.")
    return results, metadata

if __name__ == "__main__":
    train_and_evaluate()
