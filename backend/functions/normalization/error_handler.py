"""
TASK-011: Data Normalization Error Handler and Dead-Letter Pipeline Lambda
Target: backend/functions/normalization/error_handler.py

Trigger: SQS Event Source Mapping attached to IngestionDeadLetterQueue
Features:
- Consumes poison pill messages that failed 3 retry attempts in the ingestion queue.
- Parses SQS message envelope, extracts S3 bucket/key, invocation error attributes, and tenant context.
- Formats structured failure audit logs into single-table DynamoDB:
    PK: TENANT#<tenant_id>
    SK: JOB#<job_id>
    GSI1-PK: TENANT#<tenant_id>#JOBS
    GSI1-SK: <timestamp_iso>
- Updates job status to 'FAILED' with explicit standardized error codes:
    e.g. SCHEMA_VALIDATION_FAILURE, CORRUPTED_STREAM, UNRECOVERABLE_PARSING_ERROR.
- Preserves failure stack traces and dead-letter message metadata for root-cause diagnosis.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

import boto3
from botocore.config import Config

AWS_REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "ap-south-1"))
DYNAMODB_TABLE_NAME = os.environ.get("CORE_TABLE_NAME", "RevenueOS_Core_dev")

dynamodb_resource = boto3.resource("dynamodb", region_name=AWS_REGION)
table = dynamodb_resource.Table(DYNAMODB_TABLE_NAME)


def parse_dlq_message(record: Dict[str, Any]) -> Dict[str, Any]:
    body_raw = record.get("body", "{}")
    message_id = record.get("messageId", "unknown_msg")
    attributes = record.get("attributes", {})
    approximate_receive_count = attributes.get("ApproximateReceiveCount", "3")

    try:
        body = json.loads(body_raw) if isinstance(body_raw, str) else body_raw
    except Exception:
        body = {"raw_content": body_raw}

    # Extract S3 object details or direct payload
    detail = body.get("detail", {})
    bucket = detail.get("bucket", {}).get("name") or body.get("bucket", "unknown_bucket")
    key = detail.get("object", {}).get("key") or body.get("key", "unknown_key")

    # Extract tenant_id and job_id from S3 key format: <tenant_id>/csv/<job_id>.csv
    key_parts = key.split("/")
    tenant_id = body.get("tenant_id") or (key_parts[0] if len(key_parts) > 1 else "unknown_tenant")
    job_id = body.get("job_id") or (key_parts[-1].split(".")[0] if len(key_parts) > 0 else f"job_{message_id}")

    error_code = body.get("error_code") or "INGESTION_PIPELINE_FAILURE"
    error_message = body.get("error_message") or "Message routed to Dead Letter Queue after maximum retries exceeded."

    return {
        "tenant_id": tenant_id,
        "job_id": job_id,
        "message_id": message_id,
        "bucket": bucket,
        "key": key,
        "receive_count": int(approximate_receive_count) if str(approximate_receive_count).isdigit() else 3,
        "error_code": error_code,
        "error_message": error_message,
        "raw_payload": body,
    }


def record_job_failure(failure_info: Dict[str, Any]) -> None:
    now_iso = datetime.now(timezone.utc).isoformat()
    tenant_id = failure_info["tenant_id"]
    job_id = failure_info["job_id"]

    item = {
        "PK": f"TENANT#{tenant_id}",
        "SK": f"JOB#{job_id}",
        "GSI1-PK": f"TENANT#{tenant_id}#JOBS",
        "GSI1-SK": now_iso,
        "tenant_id": tenant_id,
        "job_id": job_id,
        "status": "FAILED",
        "error_code": failure_info["error_code"],
        "error_message": failure_info["error_message"],
        "retry_attempts": failure_info["receive_count"],
        "s3_bucket": failure_info["bucket"],
        "s3_key": failure_info["key"],
        "message_id": failure_info["message_id"],
        "failed_at": now_iso,
    }

    table.put_item(Item=item)


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    records = event.get("Records", [])
    processed_failures: List[Dict[str, Any]] = []

    for record in records:
        failure_info = parse_dlq_message(record)
        record_job_failure(failure_info)
        processed_failures.append({
            "tenant_id": failure_info["tenant_id"],
            "job_id": failure_info["job_id"],
            "status": "FAILED_AUDITED",
        })

    return {
        "statusCode": 200,
        "processed_dlq_count": len(processed_failures),
        "failures": processed_failures,
    }
