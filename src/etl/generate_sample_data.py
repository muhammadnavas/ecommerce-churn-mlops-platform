"""
Data Generator for E-Commerce Online Retail Dataset.
Generates realistic transaction records split across simulated monthly batches,
including valid orders, cancellations ('C' prefix), and corrupt records to test ETL rejection logging.
"""

import os
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def generate_ecommerce_dataset(output_dir="data/raw", num_customers=350, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Pre-defined catalog of e-commerce products
    products = [
        ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 2.55, "Home Decor"),
        ("71053", "WHITE METAL LANTERN", 3.39, "Home Decor"),
        ("84406B", "CREAM CUPID HEARTS COAT HANGER", 2.75, "Home Decor"),
        ("84029G", "KNITTED UNION FLAG HOT WATER BOTTLE", 3.39, "Home Accessories"),
        ("84029E", "RED WOOLLY HOTTIE WHITE HEART", 3.39, "Home Accessories"),
        ("22752", "SET 7 BABUSHKA NESTING BOXES", 7.65, "Kitchenware"),
        ("21730", "GLASS STAR FROSTED T-LIGHT HOLDER", 4.25, "Home Decor"),
        ("22633", "HAND WARMER UNION JACK", 1.85, "Accessories"),
        ("22632", "HAND WARMER RED RETROSPOT", 1.85, "Accessories"),
        ("84879", "ASSORTED COLOUR BIRD ORNAMENT", 1.69, "Home Decor"),
        ("22745", "POPPY'S PLAYHOUSE BEDROOM", 2.10, "Toys"),
        ("22748", "POPPY'S PLAYHOUSE KITCHEN", 2.10, "Toys"),
        ("22749", "FELTCRAFT PRINCESS CHARLOTTE DOLL", 3.75, "Toys"),
        ("22310", "IVORY KNITTED MUG COSY", 1.65, "Kitchenware"),
        ("84969", "BOX OF 6 ASSORTED COLOUR TEASPOONS", 4.25, "Kitchenware"),
        ("22623", "BOX OF RETRO COFFEE MUGS", 9.95, "Kitchenware"),
        ("22622", "SET OF 4 ROSE BOTANICAL CANDLES", 4.95, "Home Decor"),
        ("21754", "HOME BUILDING BLOCK WORD", 5.95, "Home Decor"),
        ("21755", "LOVE BUILDING BLOCK WORD", 5.95, "Home Decor"),
        ("21777", "RECIPE BOX PANTRY APPLE RECIPE", 2.95, "Kitchenware"),
        ("47566", "PARTY BUNTING", 4.95, "Party Supplies"),
        ("22960", "JAM MAKING SET WITH JARS", 4.25, "Kitchenware"),
        ("22961", "JAM MAKING SET PRINTED", 1.45, "Kitchenware"),
        ("23245", "SET OF 3 REGENCY CAKE STANDS", 12.75, "Kitchenware"),
        ("21931", "JUMBO BAG RED RETROSPOT", 2.08, "Bags"),
        ("22411", "JUMBO BAG VINTAGE LEAF", 2.08, "Bags"),
        ("22386", "JUMBO BAG PINK POLKADOT", 2.08, "Bags"),
        ("21929", "JUMBO BAG WHITE RETROSPOT", 2.08, "Bags"),
        ("22423", "REGENCY CAKESTAND 3 TIER", 12.75, "Kitchenware"),
        ("85099B", "JUMBO BAG RED RETROSPOT", 2.08, "Bags")
    ]
    
    countries = ["United Kingdom", "United Kingdom", "United Kingdom", "United Kingdom", 
                 "France", "Germany", "EIRE", "Spain", "Netherlands", "Belgium"]
    
    customer_ids = [12000 + i for i in range(num_customers)]
    
    # Assign customer churn probability profiles (active vs churned)
    customer_profiles = {}
    for cid in customer_ids:
        is_churner = random.random() < 0.35  # ~35% churn rate
        customer_profiles[cid] = {
            "is_churner": is_churner,
            "preferred_country": random.choice(countries),
            "activity_end_month": random.randint(1, 7) if is_churner else 12
        }
        
    start_date = datetime(2010, 12, 1)
    months = [
        ("2010_12", datetime(2010, 12, 1), datetime(2010, 12, 31)),
        ("2011_01", datetime(2011, 1, 1), datetime(2011, 1, 31)),
        ("2011_02", datetime(2011, 2, 1), datetime(2011, 2, 28)),
        ("2011_03", datetime(2011, 3, 1), datetime(2011, 3, 31)),
        ("2011_04", datetime(2011, 4, 1), datetime(2011, 4, 30)),
        ("2011_05", datetime(2011, 5, 1), datetime(2011, 5, 31)),
        ("2011_06", datetime(2011, 6, 1), datetime(2011, 6, 30)),
        ("2011_07", datetime(2011, 7, 1), datetime(2011, 7, 31)),
        ("2011_08", datetime(2011, 8, 1), datetime(2011, 8, 31)),
        ("2011_09", datetime(2011, 9, 1), datetime(2011, 9, 30)),
        ("2011_10", datetime(2011, 10, 1), datetime(2011, 10, 31)),
        ("2011_11", datetime(2011, 11, 1), datetime(2011, 11, 30))
    ]
    
    invoice_seq = 536365
    total_records = 0
    
    for month_idx, (batch_name, m_start, m_end) in enumerate(months, start=1):
        records = []
        days_in_month = (m_end - m_start).days
        
        # Determine active customers this month
        active_customers = [
            cid for cid, profile in customer_profiles.items()
            if month_idx <= profile["activity_end_month"]
        ]
        
        # Generate 150-300 invoices per month
        num_invoices = random.randint(150, 300)
        for _ in range(num_invoices):
            cid = random.choice(active_customers)
            c_country = customer_profiles[cid]["preferred_country"]
            inv_date = m_start + timedelta(
                days=random.randint(0, days_in_month),
                hours=random.randint(8, 18),
                minutes=random.randint(0, 59)
            )
            
            # Decide if normal invoice or cancellation
            is_cancel = random.random() < 0.08
            inv_str = f"C{invoice_seq}" if is_cancel else str(invoice_seq)
            invoice_seq += 1
            
            # Order items (1 to 6 items per invoice)
            num_items = random.randint(1, 6)
            sampled_products = random.sample(products, num_items)
            
            for stock_code, desc, unit_price, _cat in sampled_products:
                qty = random.randint(1, 24)
                if is_cancel:
                    qty = -abs(qty)
                    
                records.append({
                    "InvoiceNo": inv_str,
                    "StockCode": stock_code,
                    "Description": desc,
                    "Quantity": qty,
                    "InvoiceDate": inv_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "UnitPrice": unit_price,
                    "CustomerID": cid,
                    "Country": c_country
                })
        
        # Inject ~15 corrupted/invalid records per batch to demonstrate data quality checks
        for _ in range(12):
            bad_type = random.choice(["missing_customer", "invalid_price", "corrupt_date", "zero_quantity"])
            inv_str = str(invoice_seq)
            invoice_seq += 1
            stock_code, desc, unit_price, _ = random.choice(products)
            inv_date_str = m_start.strftime("%Y-%m-%d %H:%M:%S")
            
            if bad_type == "missing_customer":
                records.append({
                    "InvoiceNo": inv_str, "StockCode": stock_code, "Description": desc,
                    "Quantity": 5, "InvoiceDate": inv_date_str, "UnitPrice": unit_price,
                    "CustomerID": None, "Country": "United Kingdom"
                })
            elif bad_type == "invalid_price":
                records.append({
                    "InvoiceNo": inv_str, "StockCode": stock_code, "Description": desc,
                    "Quantity": 3, "InvoiceDate": inv_date_str, "UnitPrice": -5.00,
                    "CustomerID": random.choice(customer_ids), "Country": "United Kingdom"
                })
            elif bad_type == "corrupt_date":
                records.append({
                    "InvoiceNo": inv_str, "StockCode": stock_code, "Description": desc,
                    "Quantity": 2, "InvoiceDate": "INVALID_DATE_FORMAT", "UnitPrice": unit_price,
                    "CustomerID": random.choice(customer_ids), "Country": "United Kingdom"
                })
            elif bad_type == "zero_quantity":
                records.append({
                    "InvoiceNo": inv_str, "StockCode": stock_code, "Description": desc,
                    "Quantity": 0, "InvoiceDate": inv_date_str, "UnitPrice": unit_price,
                    "CustomerID": random.choice(customer_ids), "Country": "United Kingdom"
                })
        
        df_batch = pd.DataFrame(records)
        batch_filename = os.path.join(output_dir, f"batch_{batch_name}.csv")
        df_batch.to_csv(batch_filename, index=False)
        total_records += len(df_batch)
        print(f"Generated {batch_filename}: {len(df_batch)} records.")
        
    print(f"Dataset generation complete. Total records generated: {total_records}")
    return total_records

if __name__ == "__main__":
    generate_ecommerce_dataset()
