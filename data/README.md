---
noteId: "9e9af0105a0011f190395d1f402f7da9"
tags: []

---

# Data

The raw CSV (`creditcard_fraud.csv`, 144 MB) is **not committed to git**.
It lives in S3 and is pulled from there at training time.

## Getting the data

**Option A — Upload to your S3 feature store (recommended)**
```bash
python scripts/upload_data.py \
    --bucket $FEATURE_STORE_BUCKET \
    --data-path data/creditcard_fraud.csv
```

**Option B — Download from Kaggle**
```bash
pip install kaggle
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip -d data/
```
Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Schema

| Column | Type | Description |
|--------|------|-------------|
| Time | float | Seconds elapsed since first transaction |
| V1–V28 | float | PCA-transformed features (anonymised) |
| Amount | float | Transaction amount (USD) |
| Class | int | 0 = legitimate, 1 = fraud |

Rows: 284,807 — Class distribution: 99.83% legit / 0.17% fraud
