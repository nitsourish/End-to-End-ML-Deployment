#!/usr/bin/env bash
# =============================================================================
#  One-time App Runner infrastructure setup.
#
#  Run this ONCE with admin/SSO credentials to create:
#    1. IAM role  — lets App Runner pull images from ECR
#    2. App Runner service — the running FastAPI container
#
#  After this script completes, copy APPRUNNER_SERVICE_ARN to GitHub Secrets.
#  All future deploys use deploy.yml (aws apprunner update-service).
#
#  Prerequisites:
#    - Admin AWS credentials active (aws sso login OR aws configure)
#    - ECR repo "fraud-detection" must exist with a "latest" image
#    - MLFLOW_TRACKING_URI and FEATURE_STORE_BUCKET must be set
#
#  Usage:
#    export MLFLOW_TRACKING_URI=http://54.174.104.67:5000
#    export FEATURE_STORE_BUCKET=fraud-detection-store-sourishdey
#    export AWS_DEFAULT_REGION=us-east-1
#    bash scripts/setup_apprunner.sh
# =============================================================================
set -euo pipefail

: "${MLFLOW_TRACKING_URI:?Set MLFLOW_TRACKING_URI (e.g. http://54.174.104.67:5000)}"
: "${FEATURE_STORE_BUCKET:?Set FEATURE_STORE_BUCKET}"
: "${AWS_DEFAULT_REGION:=us-east-1}"

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_DEFAULT_REGION}.amazonaws.com/fraud-detection:latest"

# Strip any accidental trailing newlines from env vars (common copy-paste artifact)
MLFLOW_TRACKING_URI=$(printf '%s' "${MLFLOW_TRACKING_URI}" | tr -d '\r\n')
FEATURE_STORE_BUCKET=$(printf '%s' "${FEATURE_STORE_BUCKET}" | tr -d '\r\n')
APP_NAME="fraud-detection"
ROLE_NAME="${APP_NAME}-apprunner-ecr-role"
SERVICE_NAME="${APP_NAME}-api"

echo "=== App Runner One-Time Setup ==="
echo "  Account  : ${ACCOUNT_ID}"
echo "  Region   : ${AWS_DEFAULT_REGION}"
echo "  ECR image: ${ECR_URI}"
echo "  MLflow   : ${MLFLOW_TRACKING_URI}"
echo ""

# ── 1. IAM role for App Runner → ECR ─────────────────────────────────────────
echo "[1/3] Creating IAM role for App Runner ECR access …"

ROLE_ARN=$(aws iam create-role \
  --role-name "${ROLE_NAME}" \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "build.apprunner.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }' \
  --query 'Role.Arn' --output text 2>/dev/null \
  || aws iam get-role --role-name "${ROLE_NAME}" \
     --query 'Role.Arn' --output text)

aws iam attach-role-policy \
  --role-name "${ROLE_NAME}" \
  --policy-arn "arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess" \
  2>/dev/null || true

echo "  Role ARN: ${ROLE_ARN}"
sleep 10   # IAM propagation

# ── 2. Create App Runner service ──────────────────────────────────────────────
echo "[2/3] Creating App Runner service '${SERVICE_NAME}' …"

# Write config to a temp file — avoids shell-quoting issues with URLs/slashes
# and ensures newlines in values never corrupt the JSON.
cat > /tmp/apprunner_create_config.json <<APPCFG
{
  "AuthenticationConfiguration": {
    "AccessRoleArn": "${ROLE_ARN}"
  },
  "AutoDeploymentsEnabled": false,
  "ImageRepository": {
    "ImageIdentifier": "${ECR_URI}",
    "ImageRepositoryType": "ECR",
    "ImageConfiguration": {
      "Port": "8000",
      "RuntimeEnvironmentVariables": {
        "MLFLOW_TRACKING_URI": "${MLFLOW_TRACKING_URI}",
        "FEATURE_STORE_BUCKET": "${FEATURE_STORE_BUCKET}",
        "MODEL_NAME": "fraud-detection-lr",
        "MODEL_STAGE": "staging",
        "FRAUD_THRESHOLD": "0.5",
        "AWS_ACCESS_KEY_ID": "${AWS_ACCESS_KEY_ID}",
        "AWS_SECRET_ACCESS_KEY": "${AWS_SECRET_ACCESS_KEY}",
        "AWS_DEFAULT_REGION": "${AWS_DEFAULT_REGION}"
      }
    }
  }
}
APPCFG

SERVICE_ARN=$(aws apprunner create-service \
  --service-name "${SERVICE_NAME}" \
  --source-configuration "$(cat /tmp/apprunner_create_config.json)" \
  --instance-configuration '{
    "Cpu": "0.25 vCPU",
    "Memory": "0.5 GB"
  }' \
  --query 'Service.ServiceArn' --output text)

echo "  Service ARN: ${SERVICE_ARN}"

# ── 3. Wait for service to reach RUNNING ─────────────────────────────────────
echo "[3/3] Waiting for service to start (this takes ~2 minutes) …"
for i in $(seq 1 20); do
  STATUS=$(aws apprunner describe-service \
    --service-arn "${SERVICE_ARN}" \
    --query 'Service.Status' --output text)
  echo "  [${i}/20] Status: ${STATUS}"
  if [ "${STATUS}" = "RUNNING" ]; then break; fi
  if [ "${STATUS}" = "CREATE_FAILED" ]; then
    echo "❌  Service creation failed. Check App Runner console for details."
    exit 1
  fi
  sleep 30
done

SERVICE_URL=$(aws apprunner describe-service \
  --service-arn "${SERVICE_ARN}" \
  --query 'Service.ServiceUrl' --output text)

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "✅  App Runner service created!"
echo ""
echo "════════════════════════════════════════════════════════"
echo "  Service URL : https://${SERVICE_URL}"
echo "  Service ARN : ${SERVICE_ARN}"
echo ""
echo "  Add these to GitHub Secrets:"
echo "    APPRUNNER_SERVICE_ARN    = ${SERVICE_ARN}"
echo "    AWS_ACCOUNT_ID           = ${ACCOUNT_ID}"
echo "    MLFLOW_TRACKING_URI      = ${MLFLOW_TRACKING_URI}"
echo "    FEATURE_STORE_BUCKET     = ${FEATURE_STORE_BUCKET}"
echo "    AWS_ACCESS_KEY_ID        = (your AKIA key — no trailing newline!)"
echo "    AWS_SECRET_ACCESS_KEY    = (your AKIA secret)"
echo "    AWS_REGION               = ${AWS_DEFAULT_REGION}"
echo "════════════════════════════════════════════════════════"
echo ""
echo "  Smoke test:"
echo "    curl https://${SERVICE_URL}/health"
echo "    curl https://${SERVICE_URL}/ready"
