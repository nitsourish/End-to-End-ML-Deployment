"""
One-time bootstrap script: create S3 bucket + upload raw data.

Usage:
    python scripts/upload_data.py \
        --bucket my-fraud-bucket \
        --data-path data/creditcard_fraud.csv \
        [--region us-east-1] \
        [--overwrite]

This is idempotent: re-running skips the upload if the file already exists
(unless --overwrite is passed).
"""

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from src.feature_pipeline.feature_store import ensure_bucket, S3FeatureStore

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Bootstrap S3 bucket + upload raw data")
    parser.add_argument(
        "--bucket",
        default=os.getenv("FEATURE_STORE_BUCKET"),
        help="S3 bucket name (or set FEATURE_STORE_BUCKET env var)",
    )
    parser.add_argument(
        "--data-path",
        default=str(ROOT / "data" / "creditcard_fraud.csv"),
        help="Local path to the raw CSV",
    )
    parser.add_argument("--region", default=os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-upload even if the file already exists in S3")
    args = parser.parse_args()

    if not args.bucket:
        parser.error("--bucket is required (or set FEATURE_STORE_BUCKET)")

    if not Path(args.data_path).exists():
        parser.error(f"Data file not found: {args.data_path}")

    # 1. Create bucket (idempotent)
    logger.info("Ensuring bucket s3://%s exists …", args.bucket)
    ensure_bucket(args.bucket, region=args.region)

    # 2. Upload raw data
    fs = S3FeatureStore(args.bucket, region=args.region)
    uri = fs.upload_raw_data(args.data_path, overwrite=args.overwrite)
    logger.info("✅  Raw data available at %s", uri)

    # 3. Quick connectivity check
    if fs.ping():
        logger.info("✅  Bucket s3://%s is accessible", args.bucket)
    else:
        logger.error("❌  Cannot reach s3://%s — check credentials", args.bucket)
        sys.exit(1)

    print(f"\nDone. Raw data URI: {uri}")
    print(f"Next step: python -m src.training.train --s3-bucket {args.bucket}")


if __name__ == "__main__":
    main()
