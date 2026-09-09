"""
Streamlit Business Intelligence & MLOps Interactive Dashboard.
Features 6 dedicated tabs:
1. Executive Overview & Sales Trends
2. Product & Category Analytics
3. Customer Segmentation (RFM Analysis)
4. Geographical Sales Analysis
5. Customer Retention & Cohort Analysis
6. ML Churn Predictor & Model Monitoring
"""

import os
import json
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib
import yaml

# Streamlit Page Config
st.set_page_config(
    page_title="E-Commerce Sales Analytics & Churn Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stMetric {
        background: linear-gradient(135deg, #1E2640 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .metric-container {
        display: flex;
        justify-content: space-between;
        gap: 12px;
    }
    .badge-high {
        background-color: #EF4444;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
    }
    .badge-medium {
        background-color: #F59E0B;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
    }
    .badge-low {
        background-color: #10B981;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

@st.cache_data(ttl=60)
def load_warehouse_data(db_path="data/warehouse/ecommerce_dw.db"):
    target_path = Path(db_path)
    if not target_path.exists():
        target_path = BASE_DIR / db_path
    if not target_path.exists():
        return None, None, None, None, None
        
    conn = sqlite3.connect(str(target_path))
    fact_sales = pd.read_sql("SELECT * FROM fact_sales", conn)
    dim_customer = pd.read_sql("SELECT * FROM dim_customer", conn)
    dim_product = pd.read_sql("SELECT * FROM dim_product", conn)
    dim_date = pd.read_sql("SELECT * FROM dim_date", conn)
    fact_rfm = pd.read_sql("SELECT * FROM fact_customer_rfm", conn)
    dim_geo = pd.read_sql("SELECT * FROM dim_geography", conn)
    conn.close()
    
    # Merge country names into sales
    fact_sales = pd.merge(fact_sales, dim_geo, on="country_id", how="left")
    return fact_sales, dim_customer, dim_product, fact_rfm, dim_date

def load_ml_artifacts(models_dir="models"):
    target_dir = Path(models_dir)
    if not target_dir.exists():
        target_dir = BASE_DIR / models_dir
        
    model_path = target_dir / "best_model.pkl"
    scaler_path = target_dir / "scaler.pkl"
    meta_path = target_dir / "model_metadata.json"
    
    model = joblib.load(str(model_path)) if model_path.exists() else None
    scaler = joblib.load(str(scaler_path)) if scaler_path.exists() else None
    meta = None
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
            
    return model, scaler, meta

# Header
st.title("🛒 E-Commerce Sales Analytics & Customer Churn Platform")
st.markdown("---")

fact_sales, dim_customer, dim_product, fact_rfm, dim_date = load_warehouse_data()
model, scaler, meta = load_ml_artifacts()

if fact_sales is None or len(fact_sales) == 0:
    st.error("⚠️ Data Warehouse DB not found or empty! Please run the ETL pipeline first (`python src/orchestration/runner.py`).")
    st.stop()

# Sidebar Controls
st.sidebar.header("⚙️ Dashboard Controls")
available_countries = ["All"] + sorted(list(fact_sales["country_name"].dropna().unique()))
selected_country = st.sidebar.selectbox("Filter Country", available_countries)

if selected_country != "All":
    filtered_sales = fact_sales[fact_sales["country_name"] == selected_country]
else:
    filtered_sales = fact_sales.copy()

# Tabs Definition
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Executive Overview",
    "📦 Product Analytics",
    "👥 Customer RFM",
    "🌍 Geo Distribution",
    "🔄 Retention & Cohorts",
    "🤖 MLOps Churn Predictor"
])

# ----------------------------------------------------
# TAB 1: EXECUTIVE OVERVIEW
# ----------------------------------------------------
with tab1:
    st.header("Executive Summary & Sales Performance")
    
    total_rev = filtered_sales[filtered_sales["is_cancelled"] == 0]["total_amount"].sum()
    total_orders = filtered_sales["invoice_no"].nunique()
    total_custs = filtered_sales["customer_id"].nunique()
    avg_order_val = total_rev / total_orders if total_orders > 0 else 0
    cancel_count = filtered_sales[filtered_sales["is_cancelled"] == 1]["invoice_no"].nunique()
    cancel_rate = (cancel_count / total_orders) * 100 if total_orders > 0 else 0
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("💰 Total Revenue", f"£{total_rev:,.2f}")
    c2.metric("📦 Total Orders", f"{total_orders:,}")
    c3.metric("👥 Active Customers", f"{total_custs:,}")
    c4.metric("📈 Avg Order Value", f"£{avg_order_val:,.2f}")
    c5.metric("🚫 Cancellation Rate", f"{cancel_rate:.1f}%")
    
    st.markdown("### Revenue & Order Volume Trends over Time")
    filtered_sales["InvoiceMonth"] = filtered_sales["invoice_date"].str.slice(0, 7)
    monthly_trend = filtered_sales[filtered_sales["is_cancelled"] == 0].groupby("InvoiceMonth").agg(
        Revenue=("total_amount", "sum"),
        OrderCount=("invoice_no", "nunique")
    ).reset_index()
    
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Bar(
        x=monthly_trend["InvoiceMonth"], y=monthly_trend["Revenue"],
        name="Monthly Revenue (£)", marker_color="#6366F1"
    ))
    fig_trend.add_trace(go.Scatter(
        x=monthly_trend["InvoiceMonth"], y=monthly_trend["OrderCount"],
        name="Order Count", yaxis="y2", line=dict(color="#10B981", width=3)
    ))
    fig_trend.update_layout(
        template="plotly_dark",
        title="Monthly Revenue (£) and Order Volume",
        yaxis=dict(title="Revenue (£)"),
        yaxis2=dict(title="Order Count", overlaying="y", side="right"),
        legend=dict(x=0.01, y=0.99)
    )
    st.plotly_chart(fig_trend, use_container_width=True)

# ----------------------------------------------------
# TAB 2: PRODUCT ANALYTICS
# ----------------------------------------------------
with tab2:
    st.header("Product Performance & Category Distribution")
    
    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        st.subheader("Top 10 Products by Revenue")
        top_rev = filtered_sales[filtered_sales["is_cancelled"] == 0].groupby("stock_code").agg(
            Revenue=("total_amount", "sum"),
            Quantity=("quantity", "sum")
        ).reset_index().merge(dim_product, on="stock_code", how="left")
        top_rev = top_rev.sort_values(by="Revenue", ascending=False).head(10)
        
        fig_prod = px.bar(
            top_rev, x="Revenue", y="description", orientation="h",
            color="Revenue", color_continuous_scale="Viridis",
            title="Top Products by Generated Revenue"
        )
        fig_prod.update_layout(template="plotly_dark", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_prod, use_container_width=True)
        
    with col_p2:
        st.subheader("Unit Price Distribution")
        fig_hist = px.histogram(
            filtered_sales[filtered_sales["unit_price"] < 25], x="unit_price",
            nbins=30, title="Price Range Distribution (< £25)",
            color_discrete_sequence=["#F59E0B"]
        )
        fig_hist.update_layout(template="plotly_dark", xaxis_title="Unit Price (£)", yaxis_title="Item Count")
        st.plotly_chart(fig_hist, use_container_width=True)

# ----------------------------------------------------
# TAB 3: CUSTOMER SEGMENTATION (RFM)
# ----------------------------------------------------
with tab3:
    st.header("Customer Segmentation (RFM Analysis)")
    
    # Assign RFM Segments based on Recency & Monetary
    def segment_customer(row):
        if row["recency_days"] <= 30 and row["monetary"] >= 1000:
            return "VIP Champions"
        elif row["recency_days"] <= 60 and row["frequency"] >= 3:
            return "Loyal Customers"
        elif row["recency_days"] <= 90:
            return "Potential Loyalists"
        elif row["recency_days"] > 90 and row["monetary"] >= 500:
            return "At-Risk High Value"
        else:
            return "Lost / Churned"
            
    fact_rfm["Segment"] = fact_rfm.apply(segment_customer, axis=1)
    
    c_rfm1, c_rfm2 = st.columns([1.2, 1])
    
    with c_rfm1:
        st.subheader("Recency vs. Monetary Value Scatter Plot")
        fig_rfm = px.scatter(
            fact_rfm, x="recency_days", y="monetary", size="frequency",
            color="Segment", hover_data=["customer_id"],
            title="Customer Recency vs. Total Spend (Size = Frequency)",
            log_y=True, template="plotly_dark"
        )
        st.plotly_chart(fig_rfm, use_container_width=True)
        
    with c_rfm2:
        st.subheader("Customer Segment Distribution")
        seg_counts = fact_rfm["Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        fig_pie = px.pie(
            seg_counts, names="Segment", values="Count",
            hole=0.4, title="Customer Base Segment Share",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_pie.update_layout(template="plotly_dark")
        st.plotly_chart(fig_pie, use_container_width=True)

# ----------------------------------------------------
# TAB 4: GEOGRAPHICAL SALES ANALYSIS
# ----------------------------------------------------
with tab4:
    st.header("Geographical Sales Distribution")
    
    geo_df = fact_sales[fact_sales["is_cancelled"] == 0].groupby("country_name").agg(
        TotalRevenue=("total_amount", "sum"),
        OrderCount=("invoice_no", "nunique"),
        CustomerCount=("customer_id", "nunique")
    ).reset_index().sort_values(by="TotalRevenue", ascending=False)
    
    c_g1, c_g2 = st.columns([1, 1])
    
    with c_g1:
        st.subheader("Top Countries by Sales Volume")
        fig_geo_bar = px.bar(
            geo_df.head(10), x="country_name", y="TotalRevenue",
            color="TotalRevenue", color_continuous_scale="Plasma",
            title="Top 10 International Markets"
        )
        fig_geo_bar.update_layout(template="plotly_dark")
        st.plotly_chart(fig_geo_bar, use_container_width=True)
        
    with c_g2:
        st.subheader("Country Breakdown Data Table")
        st.dataframe(geo_df.style.format({"TotalRevenue": "£{:,.2f}"}), height=380, use_container_width=True)

# ----------------------------------------------------
# TAB 5: RETENTION & COHORT ANALYSIS
# ----------------------------------------------------
with tab5:
    st.header("Customer Retention & Churn Analysis")
    
    churn_counts = fact_rfm["churn_label"].value_counts()
    active_custs = churn_counts.get(0, 0)
    churned_custs = churn_counts.get(1, 0)
    churn_rate = (churned_custs / (active_custs + churned_custs)) * 100 if (active_custs + churned_custs) > 0 else 0
    
    cr1, cr2, cr3 = st.columns(3)
    cr1.metric("🟢 Active Retained Customers", f"{active_custs}")
    cr2.metric("🔴 Churned Customers (> 90d Inactive)", f"{churned_custs}")
    cr3.metric("📊 Portfolio Churn Rate", f"{churn_rate:.1f}%")
    
    st.markdown("### Comparison: Active vs Churned Customer Profile")
    profile_comp = fact_rfm.groupby("churn_label").agg(
        AvgRecency=("recency_days", "mean"),
        AvgOrders=("frequency", "mean"),
        AvgSpend=("monetary", "mean"),
        AvgItemQty=("total_items", "mean"),
        AvgCancelRate=("cancellation_rate", "mean")
    ).reset_index()
    profile_comp["Status"] = profile_comp["churn_label"].map({0: "Active", 1: "Churned"})
    
    st.dataframe(profile_comp[["Status", "AvgRecency", "AvgOrders", "AvgSpend", "AvgItemQty", "AvgCancelRate"]].style.format({
        "AvgRecency": "{:.1f} days", "AvgOrders": "{:.1f}", "AvgSpend": "£{:,.2f}",
        "AvgItemQty": "{:.0f}", "AvgCancelRate": "{:.2%}"
    }), use_container_width=True)

# ----------------------------------------------------
# TAB 6: MLOPS CHURN PREDICTOR & MONITORING
# ----------------------------------------------------
with tab6:
    st.header("🤖 MLOps Churn Prediction & Model Performance")
    
    if not model or not meta:
        st.warning("⚠️ Trained model not found! Please run `python src/ml/train.py` to train and save models.")
    else:
        st.success(f"✅ Loaded Production Model: **{meta.get('best_model_name').upper()}** (ROC-AUC: {meta.get('best_roc_auc'):.4f})")
        
        col_m1, col_m2 = st.columns([1.2, 1])
        
        with col_m1:
            st.subheader("Real-Time Single Customer Risk Calculator")
            
            sc1, sc2 = st.columns(2)
            input_recency = sc1.slider("Recency (Days since last order)", 1, 365, 45)
            input_frequency = sc2.slider("Frequency (Total Completed Orders)", 1, 50, 4)
            input_monetary = sc1.number_input("Monetary (Total Spend £)", 10.0, 10000.0, 450.0)
            input_aov = sc2.number_input("Average Order Value (£)", 5.0, 2000.0, 112.5)
            input_items = sc1.number_input("Total Items Purchased", 1, 5000, 85)
            input_cancel_rate = sc2.slider("Cancellation Rate", 0.0, 1.0, 0.05)
            input_products = sc1.slider("Distinct Products Bought", 1, 100, 12)
            input_months = sc2.slider("Active Purchase Months", 1, 12, 3)
            
            feat_cols = meta["feature_cols"]
            input_df = pd.DataFrame([{
                "recency_days": input_recency,
                "frequency": input_frequency,
                "monetary": input_monetary,
                "avg_order_value": input_aov,
                "total_items": input_items,
                "cancellation_rate": input_cancel_rate,
                "distinct_products": input_products,
                "active_months": input_months
            }])[feat_cols]
            
            if st.button("🔮 Predict Customer Churn Risk", type="primary"):
                if meta["best_model_name"] == "logistic_regression" and scaler:
                    X_eval = scaler.transform(input_df)
                else:
                    X_eval = input_df
                    
                prob = float(model.predict_proba(X_eval)[0, 1])
                pred = int(model.predict(X_eval)[0])
                
                st.markdown("---")
                res_c1, res_c2 = st.columns(2)
                res_c1.metric("Predicted Churn Probability", f"{prob:.1%}")
                
                if prob >= 0.70:
                    res_c2.markdown("#### Risk Level: <span class='badge-high'>CRITICAL CHURN RISK</span>", unsafe_allow_html=True)
                    st.error("🚨 Recommendation: Send urgent win-back promotional coupon (20% off) and personalized account review.")
                elif prob >= 0.40:
                    res_c2.markdown("#### Risk Level: <span class='badge-medium'>MODERATE CHURN RISK</span>", unsafe_allow_html=True)
                    st.warning("⚠️ Recommendation: Include in re-engagement email sequence.")
                else:
                    res_c2.markdown("#### Risk Level: <span class='badge-low'>LOW RISK (ACTIVE)</span>", unsafe_allow_html=True)
                    st.success("🎉 Recommendation: Customer is highly engaged. Recommend VIP loyalty enrollment.")

        with col_m2:
            st.subheader("Model Evaluation Comparison")
            results_dict = meta.get("results", {})
            df_res = pd.DataFrame(results_dict).T.reset_index()
            df_res.columns = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
            
            fig_model_comp = px.bar(
                df_res, x="Model", y="ROC-AUC", color="Model",
                text_auto=".3f", title="ML Model Comparison (ROC-AUC)",
                template="plotly_dark"
            )
            st.plotly_chart(fig_model_comp, use_container_width=True)
            
            st.markdown("### Lookup Existing Customer Risk")
            search_cid = st.selectbox("Select Customer ID", fact_rfm["customer_id"].unique())
            cust_row = fact_rfm[fact_rfm["customer_id"] == search_cid].iloc[0]
            st.json({
                "customer_id": search_cid,
                "recency_days": int(cust_row["recency_days"]),
                "frequency": int(cust_row["frequency"]),
                "monetary": float(cust_row["monetary"]),
                "actual_churn_status": "CHURNED" if cust_row["churn_label"] == 1 else "ACTIVE"
            })
