# Streamlit Dashboard Deployment Guide

This document details the step-by-step procedure for deploying the **E-Commerce Sales Analytics & Churn Prediction Platform** Streamlit dashboard across various environments.

---

## ☁️ Option 1: Streamlit Community Cloud (Recommended Free Cloud Hosting)

Streamlit Community Cloud provides free, instant cloud hosting directly from GitHub.

### Step 1: Push Repository to GitHub
Ensure all your files (including `requirements.txt`, `data/warehouse/ecommerce_dw.db`, `models/`, and `src/dashboard/app.py`) are pushed to your GitHub repository:
```bash
git add .
git commit -m "Deploy: Update Streamlit configuration and path loading"
git push origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/).
2. Sign in with your GitHub account.
3. Click **New app**.
4. Configure the deployment settings:
   - **Repository**: `muhammadnavas/ecommerce-churn-mlops-platform` (or your repository name)
   - **Branch**: `main`
   - **Main file path**: `src/dashboard/app.py`
   - **App URL**: *(Optional custom name)*
5. Click **Deploy!**

---

## 🐳 Option 2: Docker Container Deployment

If you prefer containerized deployment on a cloud server (AWS EC2, GCP VM, DigitalOcean, Azure) or local Docker:

### Step 1: Start Docker Desktop / Daemon
Ensure Docker engine is running on your host.

### Step 2: Build & Launch Container
From the root directory (`d:\Projects\Python\ML Project`):
```bash
docker-compose -f docker/docker-compose.yml up --build -d
```

Access the deployed dashboard at:
- **Streamlit Dashboard**: `http://localhost:8501`
- **FastAPI REST API**: `http://localhost:8000`

---

## 💻 Option 3: Local / On-Premise Production Deployment

To run Streamlit continuously on local or server infrastructure:

```bash
streamlit run src/dashboard/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
```
