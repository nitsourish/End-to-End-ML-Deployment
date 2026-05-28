---
noteId: "63f72ba05a8111f190395d1f402f7da9"
tags: []

---

# Deployment Guide — End-to-End ML Pipeline

Everything is orchestrated from a single S3 bucket. This guide takes you
from zero to a live fraud-detection API with MLflow registry, Evidently
drift monitoring, and automated retraining.

---

## Architecture at a glance

```
                        S3 Bucket (single source of truth)
                   ┌────────────────────────────────────────────┐
                   │  raw/creditcard_fraud.csv                   │
                   │  features/offline/{version}/train.parquet   │
                   │  features/artifacts/{version}/scaler.joblib │
                   │  mlflow-artifacts/  ← MLflow model binaries │
                   │  reports/{date}/    ← Evidently HTML reports │
                   └────────────────────────────────────────────┘
                        ▲         ▲              ▲
                        │         │              │
              upload    │  read   │    upload    │
              raw data  │  train  │    reports   │
                        │         │              │
             scripts/   │  CI     │  retrain.yml │
             upload_data │  retrain│              │
                        │         │              │
          ┌─────────────┤  ┌──────┤   ┌──────────┤
          │ EC2          │  │ EC2  │   │ GitHub   │
          │ MLflow       │  │ MLflow│  │ Actions  │
          │ server       │  │ DB   │  │ CI/CD    │
          └─────────────┘  └──────┘  └──────────┘
                              ▼
                       App Runner (live API)
                  https://{id}.{region}.awsapprunner.com
```

---

## Prerequisites

| Tool | Check | Install |
|------|-------|---------|
| AWS CLI v2 | `aws --version` | https://aws.amazon.com/cli |
| Docker | `docker --version` | https://docker.com |
| Python 3.13 | `python3 --version` | `brew install python@3.13` |
| git | `git --version` | pre-installed on most systems |

AWS credentials must be configured:
```bash
aws configure
# AWS Access Key ID: ...
# AWS Secret Access Key: ...
# Default region: us-east-1
# Output format: json

aws sts get-caller-identity   # verify it works
```

---

## Step 1 — Clone the repo

```bash
git clone https://github.com/nitsourish/End-to-End-ML-Deployment
cd End-to-End-ML-Deployment

# Install Python dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Step 2 — Create S3 bucket + upload raw data

```bash
export FEATURE_STORE_BUCKET=fraud-detection-store-$(whoami)
export AWS_DEFAULT_REGION=us-east-1

# Create bucket, enable versioning + encryption, upload raw CSV
python scripts/upload_data.py \
    --bucket "$FEATURE_STORE_BUCKET" \
    --data-path data/creditcard_fraud.csv

# Verify
aws s3 ls s3://${FEATURE_STORE_BUCKET}/raw/
```

Expected output:
```
Loaded reference data from s3://fraud-detection-store-*/raw/creditcard_fraud.csv
✅  Raw data available at s3://fraud-detection-store-*/raw/creditcard_fraud.csv
```

---

## Step 3 — Evidently reports (presigned URLs — no public access needed)

The feature store bucket contains sensitive model data, so we keep Block Public
Access enabled. Instead of a public website endpoint, the drift workflow
automatically generates **presigned URLs** (valid 4 hours) after each run and
writes them to the GitHub Actions job summary.

**How to view a report:**
1. Go to the GitHub Actions run (retrain workflow)
2. Click the `check-drift` job → open the **Summary** tab
3. Click the presigned link — it opens the Evidently HTML report directly
   in your browser without any AWS credentials

**How to generate a presigned URL manually:**
```bash
aws s3 presign \
  "s3://${FEATURE_STORE_BUCKET}/reports/$(date +%Y-%m-%d)/<RUN_ID>/data_drift_ci.html" \
  --expires-in 14400   # 4 hours
```

**How to list all drift reports in S3:**
```bash
aws s3 ls s3://${FEATURE_STORE_BUCKET}/reports/ --recursive
```

> **Why not a public website?**  
> `s3:PutBucketPolicy` is blocked by the bucket's Block Public Access setting
> (all four flags are `true`). Changing them is safe *only* for purely public
> assets. For a bucket that also holds raw fraud data, presigned URLs are the
> correct pattern — temporary, authenticated, no ACL changes needed.

---

## Step 4 — Launch MLflow tracking server on EC2

```bash
export EC2_KEY_PAIR=my-key-pair   # name of your existing EC2 key pair
                                   # create one: aws ec2 create-key-pair ...

bash scripts/setup_mlflow_ec2.sh
```

This script:
1. Creates a security group (SSH + port 5000 from your IP + GitHub Actions)
2. Creates an IAM role so EC2 can read/write S3
3. Launches `t3.small` Amazon Linux 2023
4. Installs MLflow as a `systemd` service via UserData
5. Configures MLflow with S3 artifact store pointing at your bucket

Wait ~2 minutes for the instance to boot, then:

```bash
# Script prints this at the end — copy it:
MLFLOW_TRACKING_URI=http://<PUBLIC_IP>:5000

# Save to your local .env
echo "MLFLOW_TRACKING_URI=http://<PUBLIC_IP>:5000" >> .env

# Verify MLflow is running
curl http://<PUBLIC_IP>:5000/health
# → {"status":"ok"}

# Open the MLflow UI in your browser
open http://<PUBLIC_IP>:5000
```

---

## Step 5 — Run first training locally

This registers the initial model into MLflow and materialises features into S3.

```bash
source .env   # load MLFLOW_TRACKING_URI, FEATURE_STORE_BUCKET

python -m src.training.train \
    --data-path data/creditcard_fraud.csv \
    --experiment-name fraud-detection-lr \
    --artifacts-dir artifacts/ \
    --s3-bucket "$FEATURE_STORE_BUCKET"
```

What happens:
```
INFO  Loaded 284807 rows from data/creditcard_fraud.csv
INFO  Split → train=227845  val=56962
INFO  L1 feature selection: kept 30/30
INFO  Scaler saved locally → artifacts/scaler.joblib
INFO  Offline features saved → s3://…/features/offline/v_20240115_143022/
INFO  Artifacts uploaded → s3://…/features/artifacts/v_20240115_143022/
INFO  C=0.01  ROC-AUC=0.9715  Avg-Prec=0.7200
...
INFO  Registered model 'fraud-detection-lr'  run_id=abc123
INFO  Model v1 promoted to Staging
```

Verify in MLflow UI (`http://<EC2_IP>:5000`):
- Experiment `fraud-detection-lr` should appear
- 7 runs (6 grid-search + 1 best model)
- Model `fraud-detection-lr` registered at Staging

Verify in S3:
```bash
aws s3 ls s3://${FEATURE_STORE_BUCKET}/features/ --recursive
# features/offline/v_20240115_143022/train.parquet
# features/offline/v_20240115_143022/val.parquet
# features/offline/v_20240115_143022/metadata.json
# features/artifacts/v_20240115_143022/scaler.joblib
# features/artifacts/v_20240115_143022/selected_features.json

aws s3 ls s3://${FEATURE_STORE_BUCKET}/mlflow-artifacts/ --recursive | head -10
# mlflow-artifacts/{run_id}/artifacts/model/...
```

---

## Step 6 — Set up GitHub repository + secrets

```bash
# Push code to GitHub
git remote set-url origin https://github.com/nitsourish/End-to-End-ML-Deployment
git add .
git commit -m "Initial commit — full ML pipeline"
git push origin main
```

Add the following secrets at:
`https://github.com/nitsourish/End-to-End-ML-Deployment/settings/secrets/actions`

| Secret name | Where to get it | Example value |
|-------------|-----------------|---------------|
| `AWS_ACCESS_KEY_ID` | IAM → Your user → Security credentials | `AKIAIOSFODNN7EXAMPLE` |
| `AWS_SECRET_ACCESS_KEY` | Same as above | `wJalrXUtnFEMI/K7MDENG...` |
| `AWS_REGION` | Your chosen region | `us-east-1` |
| `AWS_ACCOUNT_ID` | `aws sts get-caller-identity --query Account` | `123456789012` |
| `MLFLOW_TRACKING_URI` | Output of `setup_mlflow_ec2.sh` | `http://1.2.3.4:5000` |
| `FEATURE_STORE_BUCKET` | Bucket you created in Step 2 | `fraud-detection-store-john` |
| `APPRUNNER_SERVICE_ARN` | Output of Step 7 below | `arn:aws:apprunner:...` |

---

## Step 7 — Bootstrap AWS infrastructure (ECR + App Runner)

```bash
export MLFLOW_TRACKING_URI=http://<EC2_IP>:5000

bash infrastructure/aws_setup.sh
```

This creates (one-time):
1. **ECR repository** `fraud-detection` — stores Docker images
2. **IAM role** `fraud-detection-apprunner-role` — App Runner → ECR access
3. **App Runner service** `fraud-detection-api` — live HTTPS endpoint

At the end it prints:
```
✅  Setup complete!
   ECR image : 123456789012.dkr.ecr.us-east-1.amazonaws.com/fraud-detection:latest
   Service   : fraud-detection-api
```

Get the service ARN for the GitHub Secret:
```bash
aws apprunner list-services \
    --query 'ServiceSummaryList[?ServiceName==`fraud-detection-api`].ServiceArn' \
    --output text
# → arn:aws:apprunner:us-east-1:123456789012:service/fraud-detection-api/abc123
```

Add `APPRUNNER_SERVICE_ARN` to GitHub Secrets (table above).

---

## Step 8 — First automated deploy (push to main)

With all secrets set, push any change to `main` to trigger `deploy.yml`:

```bash
git commit --allow-empty -m "chore: trigger first automated deploy"
git push origin main
```

Watch the pipeline at:
`https://github.com/nitsourish/End-to-End-ML-Deployment/actions`

The `deploy.yml` sequence:
```
① Download feature artifacts from S3   (~10s)
② docker build (multi-stage)           (~3 min)
③ docker push → ECR                    (~30s)
④ aws apprunner update-service         (~5s)
⑤ Poll until RUNNING                  (~3 min)
⑥ curl /health + /ready smoke test    (~5s)
```

Total: ~7 minutes for first deploy (subsequent deploys ~4 min with layer cache).

---

## Step 9 — Verify the live service

```bash
# Get your service URL
SERVICE_URL=$(aws apprunner describe-service \
    --service-arn "$APPRUNNER_SERVICE_ARN" \
    --query 'Service.ServiceUrl' --output text)

echo "API: https://${SERVICE_URL}"

# Liveness
curl https://${SERVICE_URL}/health
# {"status":"ok"}

# Readiness (model loaded)
curl https://${SERVICE_URL}/ready
# {"status":"ready","model_version":"1"}

# Model metadata
curl https://${SERVICE_URL}/model/info | python3 -m json.tool

# Single prediction
curl -X POST https://${SERVICE_URL}/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 3600, "Amount": 120.5,
    "V1": -1.36, "V2": -0.07, "V3": 2.53, "V4": 1.38,
    "V5": -0.34, "V6": 0.46,  "V7": 0.24, "V8": 0.10,
    "V9": 0.36,  "V10": 0.09, "V11": -0.55,"V12": -0.62,
    "V13": -0.99,"V14": -0.31,"V15": 1.47, "V16": -0.47,
    "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
    "V21": -0.02,"V22": 0.28, "V23": -0.11,"V24": 0.07,
    "V25": 0.13, "V26": -0.19,"V27": 0.13, "V28": -0.02
  }'
# {"fraud_probability":0.003421,"is_fraud":false,"threshold":0.5,"model_version":"1","latency_ms":2.1}
```

---

## Step 10 — Run drift check manually

Test Evidently + the retrain decision gate before waiting for Monday's cron:

```bash
# Trigger retrain.yml manually via GitHub CLI
gh workflow run retrain.yml \
    --repo nitsourish/End-to-End-ML-Deployment \
    --ref main \
    --field force_retrain=false \
    --field drift_threshold=0.2

# OR from GitHub UI:
# Actions → "Drift Check + Conditional Retrain" → "Run workflow"
```

Watch the three jobs:
```
check-drift  →  Evidently runs, reports uploaded to S3
                GitHub step summary shows drift table
                should_retrain = true/false

retrain      →  (only if drift detected)
                Loads features from S3 Parquet
                Trains 6 LR variants + best model
                Registers new version in MLflow
                Promotes to Staging

deploy-after-retrain  →  Triggers deploy.yml
                          New image built + pushed to ECR
                          App Runner updated
```

View the Evidently HTML report:
```
http://{BUCKET}.s3-website-{REGION}.amazonaws.com/reports/{YYYY-MM-DD}/{RUN_ID}/data_drift_ci.html
```

---

## Step 11 — Automated weekly schedule

The `retrain.yml` cron (`0 3 * * 1` = Mon 03:00 UTC) is already active once
the workflow file is on `main`. No further action needed.

To verify it's enabled:
```bash
gh workflow list --repo nitsourish/End-to-End-ML-Deployment
# NAME                           STATE   ID
# CI                             active  ...
# Deploy                         active  ...
# Drift Check + Conditional Retrain  active  ...
```

---

## Step 12 — Force a retrain (e.g. after new data arrives)

```bash
# Option A: GitHub CLI
gh workflow run retrain.yml \
    --repo nitsourish/End-to-End-ML-Deployment \
    --ref main \
    --field force_retrain=true

# Option B: first materialise new features, then trigger deploy
python -m src.training.train \
    --data-path data/new_data.csv \
    --s3-bucket "$FEATURE_STORE_BUCKET"

git commit --allow-empty -m "chore: new data materialised → deploy"
git push origin main
```

---

## Monitoring reference

### MLflow UI
```
http://<EC2_IP>:5000
  → Experiments → fraud-detection-lr → compare runs
  → Models → fraud-detection-lr → version history
```

### Evidently reports (S3 static site)
```
http://{BUCKET}.s3-website-{REGION}.amazonaws.com/reports/
  → {date}/{run_id}/data_drift_ci.html   ← data drift per feature
```

### App Runner logs
```bash
aws logs tail /aws/apprunner/fraud-detection-api --follow
```

### EC2 MLflow logs
```bash
ssh -i ~/.ssh/${EC2_KEY_PAIR}.pem ec2-user@<EC2_IP>
journalctl -u mlflow -f
```

---

## S3 bucket layout (after full run)

```
s3://{FEATURE_STORE_BUCKET}/
├── raw/
│   └── creditcard_fraud.csv
├── features/
│   ├── offline/
│   │   └── v_20240115_143022/
│   │       ├── train.parquet
│   │       ├── val.parquet
│   │       └── metadata.json
│   ├── artifacts/
│   │   └── v_20240115_143022/
│   │       ├── scaler.joblib
│   │       └── selected_features.json
│   ├── latest.json               ← points to newest version
│   └── latest_artifacts.json
├── mlflow-artifacts/
│   └── {experiment_id}/{run_id}/artifacts/
│       ├── model/                ← sklearn model binary
│       ├── confusion_matrix.txt
│       └── feature_config.json
└── reports/
    └── 2024-01-15/
        └── {run_id}/
            └── data_drift_ci.html
```

---

## Cost estimate (us-east-1, light usage)

| Resource | Spec | $/month |
|----------|------|---------|
| EC2 (MLflow) | t3.small, always-on | ~$15 |
| App Runner | 1 vCPU / 2GB, ~100 req/day | ~$5 |
| S3 | ~5 GB storage + requests | ~$0.50 |
| ECR | ~1 GB images | ~$0.10 |
| **Total** | | **~$21/month** |

To minimise cost when not in use:
```bash
# Stop MLflow EC2 (keeps data, no compute charge)
aws ec2 stop-instances --instance-ids <INSTANCE_ID>

# Restart it later
aws ec2 start-instances --instance-ids <INSTANCE_ID>
# (get new public IP and update MLFLOW_TRACKING_URI)
```

---

## Troubleshooting

**`/ready` returns 503**
```
Model not loaded. Check:
1. MLFLOW_TRACKING_URI is reachable from App Runner
2. Model is registered at "Staging" in MLflow
3. App Runner logs: aws logs tail /aws/apprunner/fraud-detection-api
```

**`deploy.yml` fails at "Download feature artifacts from S3"**
```
Artifacts not in S3 yet. Run Step 5 (first training) first.
```

**MLflow UI not reachable**
```bash
# Check instance is running
aws ec2 describe-instances --instance-ids <ID> --query 'Reservations[0].Instances[0].State'

# Check MLflow service
ssh ec2-user@<IP> 'systemctl status mlflow'

# Restart if needed
ssh ec2-user@<IP> 'systemctl restart mlflow'

# Check security group allows your current IP
curl https://checkip.amazonaws.com   # get your IP
aws ec2 describe-security-groups --group-names fraud-detection-mlflow-sg
```

**Drift check always shows no drift (demo data)**
```
The demo uses tail-20% of the same CSV as "production" data — so drift
will rarely trigger. In production, replace the "Download current data"
step in retrain.yml with a query against your real inference log table
in S3 / Athena / Redshift.
```
