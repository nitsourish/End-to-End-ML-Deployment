# End-to-End ML Deployment — Credit Card Fraud Detection

> **Stack**: Scikit-learn · MLflow · FastAPI · Evidently AI · AWS App Runner · GitHub Actions

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Data / Feature Layer                         │
│  creditcard_fraud.csv → feature_engineering.py                      │
│   • log(Amount+1), hour-of-day from Time                            │
│   • StandardScaler (fit on train only)                              │
│   • L1 LR feature selection → selected_features.json               │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│                      Training (src/training/train.py)                │
│   Grid-search C ∈ {0.01…5.0}  →  Best ROC-AUC model                │
│   Every run tracked in MLflow → model registered in Model Registry  │
│   Best version auto-promoted to Staging                             │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│                   MLflow Tracking + Registry                         │
│   • Experiments: fraud-detection-lr                                 │
│   • Registered model: fraud-detection-lr  (versions: Staging/Prod)  │
│   • Artifact store: local filesystem or S3                          │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│               Serving (src/serving/app.py — FastAPI)                 │
│   POST /predict          → single real-time inference               │
│   POST /predict/batch    → batch up to 1000 transactions            │
│   GET  /health /ready    → AWS liveness/readiness probes            │
│   GET  /model/info       → current model metadata                   │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│              AWS App Runner (containerised, serverless)              │
│   Docker image → ECR → App Runner auto-scales 1–25 instances        │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                   CI/CD (GitHub Actions)                             │
│   ci.yml      → lint + test + docker build (every push/PR)         │
│   deploy.yml  → retrain + ECR push + App Runner deploy (→ main)    │
│   retrain.yml → weekly drift check → conditional retrain + deploy  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│              Monitoring (src/monitoring/drift_detection.py)          │
│   Evidently DataDriftPreset  → per-feature drift p-values           │
│   TargetDriftMonitor         → prediction distribution shift        │
│   HTML + JSON reports saved to reports/                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start (Local)

### 1 — Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2 — Start MLflow tracking server

```bash
make mlflow-start          # http://localhost:5000
```

### 3 — Train the model

```bash
make train
# or:
python -m src.training.train \
  --data-path data/creditcard_fraud.csv \
  --experiment-name fraud-detection-lr
```

Explore runs in the MLflow UI at **http://localhost:5000**.

### 4 — Start the API server

```bash
make serve                 # http://localhost:8000
```

Try it:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 3600,
    "Amount": 120.5,
    "V1": -1.36, "V2": -0.07, "V3": 2.53,
    "V4": 1.38,  "V5": -0.34, "V6": 0.46,
    "V7": 0.24,  "V8": 0.10,  "V9": 0.36,
    "V10": 0.09, "V11": -0.55,"V12": -0.62,
    "V13": -0.99,"V14": -0.31,"V15": 1.47,
    "V16": -0.47,"V17": 0.21, "V18": 0.03,
    "V19": 0.40, "V20": 0.25, "V21": -0.02,
    "V22": 0.28, "V23": -0.11,"V24": 0.07,
    "V25": 0.13, "V26": -0.19,"V27": 0.13,
    "V28": -0.02
  }'
```

### 5 — Run tests

```bash
make test
```

### 6 — Run drift detection

```bash
make drift    # uses training data as both reference + current (demo)
# reports/ directory will contain HTML + JSON Evidently reports
```

---

## ML Model Details

| Item | Value |
|------|-------|
| Algorithm | Logistic Regression (L2, `saga` solver) |
| Feature selection | L1 LR (C=0.1) → non-zero coefficients |
| Class imbalance | `class_weight='balanced'` |
| Features | V1–V28 (PCA) + `log_amount` + `hour` |
| Hyperparameter search | C ∈ {0.01, 0.05, 0.1, 0.5, 1.0, 5.0} |
| Primary metric | ROC-AUC (+ Avg Precision, F1, Recall) |
| Train/Val split | 80/20 stratified |

---

## AWS Deployment

### One-time setup

```bash
cp .env.example .env       # fill in AWS credentials + MLflow URI
source .env

# Optional: host MLflow on EC2
# ssh ec2-user@<EC2-IP>
# mlflow server --host 0.0.0.0 --port 5000 \
#   --backend-store-uri s3://<bucket>/mlflow/db \
#   --default-artifact-root s3://<bucket>/mlflow/artifacts

bash infrastructure/aws_setup.sh
```

This creates:
- **ECR repository**: `fraud-detection`
- **App Runner service**: `fraud-detection-api`
- **IAM role** for App Runner → ECR access

### GitHub Secrets required

| Secret | Description |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret |
| `AWS_REGION` | e.g. `us-east-1` |
| `AWS_ACCOUNT_ID` | 12-digit account ID |
| `MLFLOW_TRACKING_URI` | e.g. `http://<EC2-IP>:5000` |
| `APPRUNNER_SERVICE_ARN` | From `aws_setup.sh` output |

---

## CI/CD Workflows

| Workflow | Trigger | What it does |
|----------|---------|--------------|
| `ci.yml` | Every push / PR | Lint → Tests → Docker smoke build |
| `deploy.yml` | Push to `main`/`master` | Retrain → ECR push → App Runner deploy |
| `retrain.yml` | Weekly (Mon 03:00 UTC) + manual | Drift check → conditional retrain + deploy |

---

## Monitoring

Evidently generates HTML drift reports in `reports/`.

| Monitor | What it checks |
|---------|---------------|
| `DataDriftMonitor` | Per-feature distribution shift (PSI / KS test) |
| `TargetDriftMonitor` | Model prediction distribution over time |

The `retrain.yml` workflow exits with code 1 if dataset drift is detected,
automatically triggering a retrain.

---

## Project Structure

```
ml_full_deploy/
├── data/
│   └── creditcard_fraud.csv
├── src/
│   ├── feature_pipeline/
│   │   └── feature_engineering.py   # transforms, scaling, L1 selection
│   ├── training/
│   │   └── train.py                 # MLflow tracking + model registry
│   ├── serving/
│   │   └── app.py                   # FastAPI real-time inference API
│   └── monitoring/
│       └── drift_detection.py       # Evidently data + concept drift
├── tests/
│   ├── test_feature_pipeline.py
│   ├── test_model.py
│   └── test_api.py
├── infrastructure/
│   ├── docker/Dockerfile            # multi-stage production image
│   ├── apprunner.yaml               # App Runner config reference
│   └── aws_setup.sh                 # one-time AWS bootstrap
├── .github/workflows/
│   ├── ci.yml                       # lint + test + docker build
│   ├── deploy.yml                   # retrain + ECR + App Runner
│   └── retrain.yml                  # scheduled drift-aware retrain
├── .env.example
├── .gitignore
├── Makefile
├── requirements.txt
├── setup.cfg
└── README.md
```
