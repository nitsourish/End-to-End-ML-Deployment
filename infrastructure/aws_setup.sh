#!/usr/bin/env bash
# =============================================================================
#  AWS one-time setup script
#  Run this ONCE to bootstrap ECR repo + App Runner service.
#  Subsequent deploys are handled by GitHub Actions.
# =============================================================================
set -euo pipefail

: "${AWS_REGION:=us-east-1}"
: "${AWS_ACCOUNT_ID:=$(aws sts get-caller-identity --query Account --output text)}"
: "${APP_NAME:=fraud-detection}"
: "${ECR_REPO:=${APP_NAME}}"
: "${APPRUNNER_SERVICE:=${APP_NAME}-api}"
: "${MLFLOW_TRACKING_URI:?Set MLFLOW_TRACKING_URI}"

IMAGE_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:latest"

echo "=== Account: ${AWS_ACCOUNT_ID}  Region: ${AWS_REGION} ==="

# ---- 1. ECR repository ---------------------------------------------------
echo "[1/4] Creating ECR repository …"
aws ecr create-repository \
    --repository-name "${ECR_REPO}" \
    --region "${AWS_REGION}" \
    --image-scanning-configuration scanOnPush=true \
    2>/dev/null || echo "  Repository already exists — skipping"

# ---- 2. Build & push image -----------------------------------------------
echo "[2/4] Building and pushing Docker image …"
aws ecr get-login-password --region "${AWS_REGION}" \
    | docker login --username AWS --password-stdin \
      "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

docker build \
    -f infrastructure/docker/Dockerfile \
    -t "${IMAGE_URI}" \
    .

docker push "${IMAGE_URI}"

# ---- 3. IAM role for App Runner ------------------------------------------
echo "[3/4] Setting up App Runner IAM role …"
ROLE_NAME="${APP_NAME}-apprunner-role"
TRUST_POLICY='{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "build.apprunner.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}'

aws iam create-role \
    --role-name "${ROLE_NAME}" \
    --assume-role-policy-document "${TRUST_POLICY}" \
    2>/dev/null || echo "  Role already exists — skipping"

aws iam attach-role-policy \
    --role-name "${ROLE_NAME}" \
    --policy-arn "arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess" \
    2>/dev/null || true

ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/${ROLE_NAME}"

# ---- 4. Create App Runner service ----------------------------------------
echo "[4/4] Creating App Runner service …"
aws apprunner create-service \
    --service-name "${APPRUNNER_SERVICE}" \
    --source-configuration "{
        \"ImageRepository\": {
            \"ImageIdentifier\": \"${IMAGE_URI}\",
            \"ImageConfiguration\": {
                \"Port\": \"8000\",
                \"RuntimeEnvironmentVariables\": {
                    \"MLFLOW_TRACKING_URI\": \"${MLFLOW_TRACKING_URI}\",
                    \"MODEL_NAME\": \"fraud-detection-lr\",
                    \"MODEL_STAGE\": \"Staging\",
                    \"FRAUD_THRESHOLD\": \"0.5\"
                }
            },
            \"ImageRepositoryType\": \"ECR\"
        },
        \"AutoDeploymentsEnabled\": false,
        \"AuthenticationConfiguration\": {
            \"AccessRoleArn\": \"${ROLE_ARN}\"
        }
    }" \
    --instance-configuration "{
        \"Cpu\": \"1 vCPU\",
        \"Memory\": \"2 GB\"
    }" \
    --health-check-configuration "{
        \"Protocol\": \"HTTP\",
        \"Path\": \"/health\",
        \"Interval\": 10,
        \"Timeout\": 5,
        \"HealthyThreshold\": 1,
        \"UnhealthyThreshold\": 5
    }" \
    --region "${AWS_REGION}" \
    2>/dev/null || echo "  Service already exists — run deploy instead"

echo ""
echo "✅  Setup complete!"
echo "   ECR image : ${IMAGE_URI}"
echo "   Service   : ${APPRUNNER_SERVICE}"
echo ""
echo "Next: push to main branch to trigger GitHub Actions CI/CD deploy."
