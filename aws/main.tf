terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "4.67.0"
    }
  }
}

provider "aws" {
  region = var.region
}

data "aws_caller_identity" "me" {}

locals {
  bucket_name = "${var.project}-${data.aws_caller_identity.me.account_id}"
  folders     = ["raw/", "processed/", "rejected/", "models/", "reports/"]
  tags = {
    project = var.project
    owner   = "nour"
  }
}

data "aws_s3_bucket" "data" {
  bucket = local.bucket_name
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket                  = data.aws_s3_bucket.data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = data.aws_s3_bucket.data.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = data.aws_s3_bucket.data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_object" "folders" {
  for_each = toset(local.folders)
  bucket   = data.aws_s3_bucket.data.id
  key      = each.value
  content  = ""
  tags     = local.tags
}