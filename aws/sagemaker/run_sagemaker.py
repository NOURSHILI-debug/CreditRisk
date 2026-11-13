"""Dry-run-first launcher for one SageMaker XGBoost baseline job."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

BUCKET = "credit-risk-498265543404"
REGION = "us-east-1"
ROLE_ARN = "arn:aws:iam::498265543404:role/LabRole"
INPUT_URI = f"s3://{BUCKET}/raw/application_train.csv"
OUTPUT_PREFIX = f"s3://{BUCKET}/reports/sagemaker_v1"
MODEL_PREFIX = f"s3://{BUCKET}/models/sagemaker_v1"
INSTANCE_TYPE = "ml.m5.large"
FRAMEWORK_VERSION = "3.2-0"
MAX_RUN_SECONDS = 1800
# us-east-1 Linux On-Demand rate; excludes storage and data transfer.
HOURLY_RATE_USD = 0.115


def job_config() -> dict[str, object]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return {
        "job_name": f"credit-risk-xgb-baseline-{timestamp}",
        "role": ROLE_ARN,
        "region": REGION,
        "input": INPUT_URI,
        "output_path": OUTPUT_PREFIX,
        "model_prefix": MODEL_PREFIX,
        "instance_type": INSTANCE_TYPE,
        "instance_count": 1,
        "max_run_seconds": MAX_RUN_SECONDS,
        "use_spot_instances": False,
        "framework": "sagemaker.xgboost.estimator.XGBoost",
        "framework_version": FRAMEWORK_VERSION,
        "entry_point": "train.py",
        "source_dir": "aws/sagemaker",
        "dependencies": ["src"],
        "tags": [{"Key": "project", "Value": "credit-risk"}],
        "estimated_cost_usd": round(
            HOURLY_RATE_USD * MAX_RUN_SECONDS / 3600, 4
        ),
    }


def launch(config: dict[str, object]) -> tuple[str, str, str]:
    # Imports are intentionally delayed so --dry-run never loads AWS SDK code.
    import boto3
    import sagemaker
    from sagemaker.inputs import TrainingInput
    from sagemaker.xgboost.estimator import XGBoost

    boto_session = boto3.Session(region_name=REGION)
    session = sagemaker.Session(boto_session=boto_session)
    estimator = XGBoost(
        entry_point="train.py",
        source_dir="aws/sagemaker",
        dependencies=["src"],
        role=ROLE_ARN,
        framework_version=FRAMEWORK_VERSION,
        py_version="py3",
        instance_count=1,
        instance_type=INSTANCE_TYPE,
        output_path=OUTPUT_PREFIX,
        max_run=MAX_RUN_SECONDS,
        use_spot_instances=False,
        base_job_name="credit-risk-xgb-baseline",
        tags=[{"Key": "project", "Value": "credit-risk"}],
        sagemaker_session=session,
    )
    estimator.fit(
        {"train": TrainingInput(INPUT_URI, content_type="text/csv")},
        job_name=str(config["job_name"]),
        wait=True,
        logs=True,
    )
    model_uri = f"{MODEL_PREFIX}/{config['job_name']}/model.tar.gz"
    metrics_uri = f"{OUTPUT_PREFIX}/{config['job_name']}/output/output.tar.gz"
    session.boto_session.resource("s3").Object(
        BUCKET, f"models/sagemaker_v1/{config['job_name']}/model.tar.gz"
    ).copy_from(
        CopySource={
            "Bucket": BUCKET,
            "Key": str(estimator.model_data).removeprefix(f"s3://{BUCKET}/"),
        }
    )
    return str(config["job_name"]), model_uri, metrics_uri


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Submit one training job.")
    mode.add_argument(
        "--dry-run", action="store_true", help="Print config only (default)."
    )
    args = parser.parse_args()
    config = job_config()
    if not args.run:
        print(json.dumps(config, indent=2))
        print(
            "S3 outputs: "
            f"model={config['model_prefix']}/{config['job_name']}/model.tar.gz; "
            f"metrics={config['output_path']}/{config['job_name']}/output/output.tar.gz"
        )
        return
    name, model_uri, metrics_uri = launch(config)
    print(f"Job name: {name}")
    print(f"Model S3 path: {model_uri}")
    print(f"Metrics S3 path: {metrics_uri}")


if __name__ == "__main__":
    main()
