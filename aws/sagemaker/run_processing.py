"""Dry-run-first SageMaker Processing launcher for the XGBoost baseline."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aws.sagemaker.run_sagemaker import (  # noqa: E402
    BUCKET,
    FRAMEWORK_IMAGE_URI,
    FRAMEWORK_VERSION,
    INPUT_URI,
    REGION,
    ROLE_ARN,
)

PROCESSOR_INSTANCE_TYPE = "ml.t3.xlarge"
PROCESSING_VOLUME_GB = 10
MAX_RUNTIME_SECONDS = 1800
# Approximate us-east-1 Processing price; estimate only, not a live quote.
HOURLY_PRICE_ESTIMATE_USD = 0.21
MODEL_OUTPUT_ROOT = f"s3://{BUCKET}/models/sagemaker_v1"
REPORT_OUTPUT_ROOT = f"s3://{BUCKET}/reports/sagemaker_v1"
PROCESSING_INPUT_DIR = "/opt/ml/processing/input"
PROCESSING_MODEL_DIR = "/opt/ml/processing/model"
PROCESSING_OUTPUT_DIR = "/opt/ml/processing/output"


def job_config(instance_type: str = PROCESSOR_INSTANCE_TYPE) -> dict[str, object]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    job_name = f"credit-risk-xgb-processing-{timestamp}"
    max_cost = HOURLY_PRICE_ESTIMATE_USD * MAX_RUNTIME_SECONDS / 3600
    return {
        "job_name": job_name,
        "role": ROLE_ARN,
        "region": REGION,
        "instance_type": instance_type,
        "instance_count": 1,
        "volume_size_in_gb": PROCESSING_VOLUME_GB,
        "max_runtime_in_seconds": MAX_RUNTIME_SECONDS,
        "image_uri": FRAMEWORK_IMAGE_URI,
        "framework_version": FRAMEWORK_VERSION,
        "command": ["python3"],
        "entry_point": "aws/sagemaker/processing_entrypoint.py",
        "source_dir": "aws/sagemaker",
        "dependencies": ["src"],
        "requirements_file": "aws/sagemaker/requirements.txt",
        "input": INPUT_URI,
        "input_destination": PROCESSING_INPUT_DIR,
        "model_output_source": PROCESSING_MODEL_DIR,
        "model_output_s3": f"{MODEL_OUTPUT_ROOT}/{job_name}/",
        "report_output_source": PROCESSING_OUTPUT_DIR,
        "report_output_s3": f"{REPORT_OUTPUT_ROOT}/{job_name}/",
        "spot": False,
        "endpoint": False,
        "tags": [{"Key": "project", "Value": "credit-risk"}],
        "estimated_hourly_price_usd": HOURLY_PRICE_ESTIMATE_USD,
        "hourly_price_is_estimate": True,
        "max_estimated_cost_usd": round(max_cost + 1e-9, 2),
        "cost_estimate_note": "Approximate compute only; S3 storage is excluded.",
    }


def launch(config: dict[str, object]) -> tuple[str, str, str]:
    # Keep --dry-run independent of AWS SDK and credential loading.
    import boto3
    import sagemaker
    from sagemaker.processing import FrameworkProcessor, ProcessingInput, ProcessingOutput
    from sagemaker.xgboost.estimator import XGBoost

    boto_session = boto3.Session(region_name=REGION)
    session = sagemaker.Session(boto_session=boto_session)
    processor = FrameworkProcessor(
        estimator_cls=XGBoost,
        framework_version=FRAMEWORK_VERSION,
        image_uri=FRAMEWORK_IMAGE_URI,
        py_version="py3",
        command=["python3"],
        role=ROLE_ARN,
        instance_type=str(config["instance_type"]),
        instance_count=1,
        volume_size_in_gb=PROCESSING_VOLUME_GB,
        max_runtime_in_seconds=MAX_RUNTIME_SECONDS,
        base_job_name="credit-risk-xgb-processing",
        tags=[{"Key": "project", "Value": "credit-risk"}],
        sagemaker_session=session,
    )
    job_name = str(config["job_name"])
    processor.run(
        code="processing_entrypoint.py",
        source_dir="aws/sagemaker",
        dependencies=["src"],
        inputs=[
            ProcessingInput(
                input_name="application-train",
                source=INPUT_URI,
                destination=PROCESSING_INPUT_DIR,
                s3_data_type="S3Prefix",
                s3_input_mode="File",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="model",
                source=PROCESSING_MODEL_DIR,
                destination=str(config["model_output_s3"]),
            ),
            ProcessingOutput(
                output_name="output",
                source=PROCESSING_OUTPUT_DIR,
                destination=str(config["report_output_s3"]),
            ),
        ],
        arguments=[
            "--data-path",
            f"{PROCESSING_INPUT_DIR}/application_train.csv",
            "--model-dir",
            PROCESSING_MODEL_DIR,
            "--output-dir",
            PROCESSING_OUTPUT_DIR,
            "--project-root",
            "/opt/ml/processing/input/code",
        ],
        job_name=job_name,
        wait=True,
        logs=True,
    )
    return (
        job_name,
        str(config["model_output_s3"]),
        str(config["report_output_s3"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Submit one Processing job.")
    mode.add_argument(
        "--dry-run", action="store_true", help="Print config only (default)."
    )
    parser.add_argument(
        "--instance-type",
        default=PROCESSOR_INSTANCE_TYPE,
        help=f"Processing instance type (default: {PROCESSOR_INSTANCE_TYPE}).",
    )
    args = parser.parse_args()
    config = job_config(instance_type=args.instance_type)
    if not args.run:
        print(json.dumps(config, indent=2))
        print(
            f"Cost estimate: ~${HOURLY_PRICE_ESTIMATE_USD:.2f}/hour (estimate); "
            f"worst case for {MAX_RUNTIME_SECONDS} seconds: "
            f"${config['max_estimated_cost_usd']:.2f} compute."
        )
        print(f"Model S3 output: {config['model_output_s3']}")
        print(f"Report S3 output: {config['report_output_s3']}")
        return
    name, model_output, report_output = launch(config)
    print(f"Processing job name: {name}")
    print(f"Model S3 output: {model_output}")
    print(f"Report S3 output: {report_output}")


if __name__ == "__main__":
    main()
