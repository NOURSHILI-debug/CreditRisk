# SageMaker XGBoost baseline

The launcher defaults to a dry run. It does not contact AWS unless `--run` is
passed. Use the AWS venv only for the launcher.

```powershell
.\.venv-aws\Scripts\python.exe aws\sagemaker\run_sagemaker.py
.\.venv-aws\Scripts\python.exe aws\sagemaker\run_sagemaker.py --run
```

The job reads `s3://credit-risk-498265543404/raw/application_train.csv`. It uses
one `ml.m5.large` instance and SageMaker XGBoost framework version `3.2-0`.
The maximum run time is 30 minutes. Spot instances are off. The estimate uses
$0.115 per instance hour and is about $0.06. S3 storage and data transfer are
not included. Actual billed compute may be lower if the job finishes early.

Model output uses `s3://credit-risk-498265543404/models/sagemaker_v1/`. Metrics
and predictions use
`s3://credit-risk-498265543404/reports/sagemaker_v1/`. Download and extract the
job's `output.tar.gz` to a local folder, then compare it:

```powershell
.\.venv\Scripts\python.exe compare_metrics.py .\downloaded-sagemaker-output
```

The comparison checks that test applicants and labels match. It reports
ROC-AUC, PR-AUC, and KS. The default tolerance is 0.005.

Set the job name printed by the launcher. List jobs and endpoints first.
Training job records cannot be deleted. Stop a job only if it is still running.

```powershell
$JobName = "credit-risk-xgb-baseline-<timestamp>"
aws sagemaker list-training-jobs --region us-east-1 --name-contains credit-risk-xgb-baseline
aws sagemaker list-endpoints --region us-east-1
aws sagemaker describe-training-job --region us-east-1 --training-job-name $JobName
```

If the job is still running, stop it. Delete its model and report artifacts from
S3 when they are no longer needed. Run these cleanup commands only after
checking the job status and confirming that the artifacts are no longer needed:

```powershell
aws sagemaker stop-training-job --region us-east-1 --training-job-name $JobName
aws s3 rm "s3://credit-risk-498265543404/models/sagemaker_v1/$JobName/" --recursive --region us-east-1
aws s3 rm "s3://credit-risk-498265543404/reports/sagemaker_v1/$JobName/" --recursive --region us-east-1
```

The stop command has no effect after the job finishes. This launcher creates
no endpoint, so the endpoint list should be empty for this baseline.
