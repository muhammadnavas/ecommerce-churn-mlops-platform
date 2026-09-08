"""
Apache Airflow DAG Definition for E-Commerce Data Engineering & MLOps Pipeline.
Orchestrates scheduled batch ingestion, data validation, star schema build, and model training.
"""

from datetime import datetime, timedelta

# Fallback dummy operators if running without Airflow package installed
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    HAS_AIRFLOW = True
except ImportError:
    HAS_AIRFLOW = False

default_args = {
    'owner': 'data_engineering_mlops',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def task_ingest():
    from src.etl.ingest import run_ingestion
    run_ingestion()

def task_clean():
    from src.etl.clean import run_cleaning
    run_cleaning()

def task_transform():
    from src.etl.transform import run_transform
    run_transform()

def task_train():
    from src.ml.train import train_and_evaluate
    train_and_evaluate()

if HAS_AIRFLOW:
    dag = DAG(
        'ecommerce_sales_and_churn_pipeline',
        default_args=default_args,
        description='End-to-end retail ETL, star schema build, and churn MLOps training',
        schedule_interval='@daily',
        catchup=False
    )

    t1 = PythonOperator(task_id='ingest_raw_batches', python_callable=task_ingest, dag=dag)
    t2 = PythonOperator(task_id='clean_and_validate_data', python_callable=task_clean, dag=dag)
    t3 = PythonOperator(task_id='build_star_schema_and_mart', python_callable=task_transform, dag=dag)
    t4 = PythonOperator(task_id='train_churn_ml_models', python_callable=task_train, dag=dag)

    t1 >> t2 >> t3 >> t4
