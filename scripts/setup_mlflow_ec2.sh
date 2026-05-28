#!/usr/bin/env bash
# =============================================================================
#  Launch + configure an EC2 instance running MLflow tracking server.
#
#  Runs from your LOCAL machine (requires AWS CLI + EC2 key pair).
#
#  What it creates:
#    - t3.small EC2 (Amazon Linux 2023)
#    - Security group: SSH (22) + MLflow UI (5000) — restricted to your IP
#    - IAM instance profile with S3 read/write for artifact store
#    - MLflow server as a systemd service (survives reboots)
#    - Artifact store: s3://{FEATURE_STORE_BUCKET}/mlflow-artifacts/
#    - Backend store: SQLite on local disk (/home/ec2-user/mlflow.db)
#
#  Usage:
#    export FEATURE_STORE_BUCKET=my-fraud-feature-store
#    export AWS_DEFAULT_REGION=us-east-1
#    export EC2_KEY_PAIR=my-key-pair          # existing EC2 key pair name
#    bash scripts/setup_mlflow_ec2.sh
# =============================================================================
set -euo pipefail

: "${FEATURE_STORE_BUCKET:?Set FEATURE_STORE_BUCKET}"
: "${EC2_KEY_PAIR:?Set EC2_KEY_PAIR (name of existing EC2 key pair)}"
: "${AWS_DEFAULT_REGION:=us-east-1}"

APP_NAME="fraud-detection"
INSTANCE_TYPE="t3.small"
AMI_ID=$(aws ec2 describe-images \
    --owners amazon \
    --filters "Name=name,Values=al2023-ami-2023*-x86_64" \
              "Name=state,Values=available" \
    --query "sort_by(Images, &CreationDate)[-1].ImageId" \
    --output text)

MY_IP=$(curl -s https://checkip.amazonaws.com)
echo "=== MLflow EC2 Setup ==="
echo "  Region  : ${AWS_DEFAULT_REGION}"
echo "  AMI     : ${AMI_ID}"
echo "  Bucket  : s3://${FEATURE_STORE_BUCKET}/mlflow-artifacts/"
echo "  Your IP : ${MY_IP}"
echo ""

# ---- 1. Security group -------------------------------------------------------
echo "[1/5] Creating security group …"
# AWS requires pure ASCII in the description — no em-dashes or other Unicode.
# We also pass --vpc-id explicitly to target the default VPC and make the
# describe fallback reliable when the SG already exists.
DEFAULT_VPC=$(aws ec2 describe-vpcs \
    --filters "Name=is-default,Values=true" \
    --query 'Vpcs[0].VpcId' --output text)

if [ -z "${DEFAULT_VPC}" ] || [ "${DEFAULT_VPC}" = "None" ]; then
    echo "ERROR: No default VPC found in region ${AWS_DEFAULT_REGION}."
    echo "  Create a default VPC first:  aws ec2 create-default-vpc"
    exit 1
fi
echo "  Using VPC: ${DEFAULT_VPC}"

SG_ID=$(aws ec2 create-security-group \
    --group-name "${APP_NAME}-mlflow-sg" \
    --description "MLflow tracking server - SSH and port 5000" \
    --vpc-id "${DEFAULT_VPC}" \
    --query 'GroupId' --output text 2>/dev/null \
    || aws ec2 describe-security-groups \
        --filters "Name=group-name,Values=${APP_NAME}-mlflow-sg" \
                  "Name=vpc-id,Values=${DEFAULT_VPC}" \
        --query 'SecurityGroups[0].GroupId' --output text)

if [ -z "${SG_ID}" ] || [ "${SG_ID}" = "None" ]; then
    echo "ERROR: Failed to create or find security group."
    echo "  Try manually: aws ec2 create-security-group --group-name ${APP_NAME}-mlflow-sg ..."
    exit 1
fi

# SSH — your IP only
aws ec2 authorize-security-group-ingress \
    --group-id "${SG_ID}" \
    --protocol tcp --port 22 \
    --cidr "${MY_IP}/32" 2>/dev/null || true

# MLflow UI — your IP only
aws ec2 authorize-security-group-ingress \
    --group-id "${SG_ID}" \
    --protocol tcp --port 5000 \
    --cidr "${MY_IP}/32" 2>/dev/null || true

# Also allow GitHub Actions IP ranges (needed for CI to log experiments)
# GitHub publishes its IP ranges; for simplicity open to 0.0.0.0/0
# In production: restrict to GitHub meta IP ranges or use VPN
aws ec2 authorize-security-group-ingress \
    --group-id "${SG_ID}" \
    --protocol tcp --port 5000 \
    --cidr "0.0.0.0/0" 2>/dev/null || true

echo "  Security group: ${SG_ID}"

# ---- 2. IAM instance profile for S3 access ----------------------------------
echo "[2/5] Creating IAM role for EC2 → S3 access …"
ROLE_NAME="${APP_NAME}-mlflow-ec2-role"

aws iam create-role \
    --role-name "${ROLE_NAME}" \
    --assume-role-policy-document '{
        "Version":"2012-10-17",
        "Statement":[{"Effect":"Allow",
                      "Principal":{"Service":"ec2.amazonaws.com"},
                      "Action":"sts:AssumeRole"}]}' \
    2>/dev/null || echo "  Role already exists"

aws iam attach-role-policy \
    --role-name "${ROLE_NAME}" \
    --policy-arn "arn:aws:iam::aws:policy/AmazonS3FullAccess" \
    2>/dev/null || true

PROFILE_NAME="${ROLE_NAME}-profile"
aws iam create-instance-profile \
    --instance-profile-name "${PROFILE_NAME}" 2>/dev/null || true
aws iam add-role-to-instance-profile \
    --instance-profile-name "${PROFILE_NAME}" \
    --role-name "${ROLE_NAME}" 2>/dev/null || true

sleep 10   # IAM propagation

# ---- 3. UserData: install + configure MLflow as systemd service -------------
echo "[3/5] Preparing UserData startup script …"
USER_DATA=$(cat <<USERDATA
#!/bin/bash
set -e
yum update -y
yum install -y python3-pip python3-devel gcc sqlite

pip3 install mlflow boto3 psutil

# Create MLflow systemd service
cat > /etc/systemd/system/mlflow.service <<EOF
[Unit]
Description=MLflow Tracking Server
After=network.target

[Service]
User=ec2-user
Restart=always
ExecStart=/usr/local/bin/mlflow server \\
    --host 0.0.0.0 \\
    --port 5000 \\
    --backend-store-uri sqlite:////home/ec2-user/mlflow.db \\
    --default-artifact-root s3://${FEATURE_STORE_BUCKET}/mlflow-artifacts/ \\
    --serve-artifacts
Environment="AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION}"
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable mlflow
systemctl start mlflow

echo "MLflow server started at http://\$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):5000"
USERDATA
)

# ---- 4. Launch EC2 instance --------------------------------------------------
echo "[4/5] Launching EC2 instance (${INSTANCE_TYPE}) …"
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "${AMI_ID}" \
    --instance-type "${INSTANCE_TYPE}" \
    --key-name "${EC2_KEY_PAIR}" \
    --security-group-ids "${SG_ID}" \
    --iam-instance-profile "Name=${PROFILE_NAME}" \
    --user-data "${USER_DATA}" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${APP_NAME}-mlflow}]" \
    --query 'Instances[0].InstanceId' --output text)

echo "  Instance ID: ${INSTANCE_ID}"
echo "  Waiting for running state …"
aws ec2 wait instance-running --instance-ids "${INSTANCE_ID}"

PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "${INSTANCE_ID}" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

# ---- 5. Output summary -------------------------------------------------------
echo ""
echo "✅  EC2 launched: ${INSTANCE_ID}"
echo "    Public IP   : ${PUBLIC_IP}"
echo "    MLflow URL  : http://${PUBLIC_IP}:5000"
echo "    (allow ~2 min for UserData to finish installing MLflow)"
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Add this to GitHub Secrets:"
echo "    MLFLOW_TRACKING_URI = http://${PUBLIC_IP}:5000"
echo ""
echo "  Add this to your local .env:"
echo "    MLFLOW_TRACKING_URI=http://${PUBLIC_IP}:5000"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "  SSH access:"
echo "    ssh -i ~/.ssh/${EC2_KEY_PAIR}.pem ec2-user@${PUBLIC_IP}"
echo ""
echo "  Check MLflow service status (after 2 min):"
echo "    ssh -i ~/.ssh/${EC2_KEY_PAIR}.pem ec2-user@${PUBLIC_IP} 'systemctl status mlflow'"
