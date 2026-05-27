# AWS App Runner — Configuration Reference

## How this project configures App Runner

This project uses **ECR image mode** (Mode B). App Runner pulls a pre-built
Docker image from ECR. There is no `apprunner.yaml` file read by the service.

All configuration is passed through the AWS CLI in `deploy.yml`:

```
aws apprunner update-service \
  --service-arn <ARN> \
  --source-configuration '{
      "ImageRepository": {
          "ImageIdentifier": "<ECR_IMAGE_URI>",
          "ImageConfiguration": {
              "Port": "8000",
              "RuntimeEnvironmentVariables": {
                  "MLFLOW_TRACKING_URI": "...",
                  "FEATURE_STORE_BUCKET": "...",
                  "MODEL_NAME": "fraud-detection-lr",
                  "MODEL_STAGE": "Staging",
                  "FRAUD_THRESHOLD": "0.5"
              }
          },
          "ImageRepositoryType": "ECR"
      },
      "AutoDeploymentsEnabled": false
  }'
```

---

## The two App Runner source modes

### Mode A — Source Code Repository (NOT used here)
App Runner connects directly to a GitHub repository via an OAuth app.
On every push it clones the repo, reads `apprunner.yaml`, builds the code,
and runs it. No Docker involved.

```yaml
# apprunner.yaml — only relevant in Mode A
version: 1.0
runtime: python311
build:
  commands:
    build:
      - pip install -r requirements.txt
run:
  command: uvicorn src.serving.app:app --host 0.0.0.0 --port 8080
  network:
    port: 8080
  env:
    - name: MODEL_NAME
      value: fraud-detection-lr
```

**When to use Mode A:** simple apps with no heavy build steps, no Docker familiarity needed.  
**Downside:** no multi-stage builds, no compiled extensions, slower startup (builds on each deploy).

### Mode B — Container Image / ECR (this project)
You build the Docker image in CI, push it to ECR, then tell App Runner
which image tag to run. Configuration is passed via API — not a file.

**When to use Mode B:** ML projects with compiled packages (scikit-learn,
scipy), multi-stage builds, baked-in artifacts, or when you need full
control over the container environment.

---

## Service settings (for reference)

| Setting | Value |
|---------|-------|
| CPU | 1 vCPU |
| Memory | 2 GB |
| Port | 8000 |
| Health check path | `/health` |
| Health check interval | 10s |
| Auto-deployments | Disabled (GitHub Actions is the only trigger) |
| Min instances | 1 |
| Max instances | 25 |
