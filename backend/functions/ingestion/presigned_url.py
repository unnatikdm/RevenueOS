"""
TASK-007: S3 Presigned URL Generator Lambda
Target: backend/functions/ingestion/presigned_url.py

Trigger: POST /uploads/presigned-url
Features:
- Extracts tenant_id strictly from Cognito JWT claims (requestContext.authorizer.jwt.claims['custom:tenant_id']).
- Validates request payload: fileName, fileType, byteSize.
- Enforces max file size limit (50 MB) and restricts MIME types to CSV and XLSX.
- Generates S3 presigned PUT URL with 15-minute expiration (900 seconds).
- Includes custom tenant metadata in the S3 upload.
- Returns JSON with uploadUrl, s3Key, and jobId.
"""

import json
import os
import uuid
from typing import Any, Dict
import boto3
from botocore.config import Config

# Limits & Allowed MIME types
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
ALLOWED_MIME_TYPES = {
    "text/csv": ".csv",
    "application/vnd.ms-excel": ".csv",
    "text/plain": ".csv",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
}

s3_client = boto3.client("s3", config=Config(signature_version="s3v4"))
INGESTION_BUCKET = os.environ.get("INGESTION_BUCKET_NAME", "revenueos-raw-dev")


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    # 1. Extract tenant_id strictly from Cognito JWT authorizer claims
    request_context = event.get("requestContext", {})
    authorizer = request_context.get("authorizer", {})
    jwt_data = authorizer.get("jwt", {})
    claims = jwt_data.get("claims", {})

    tenant_id = claims.get("custom:tenant_id") or claims.get("tenant_id")
    if not tenant_id:
        return {
            "statusCode": 401,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "Unauthorized: Missing custom:tenant_id claim in JWT token"}),
        }

    # 2. Parse request body
    body_raw = event.get("body")
    if not body_raw:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "Bad Request: Empty request body"}),
        }

    try:
        body = json.loads(body_raw) if isinstance(body_raw, str) else body_raw
    except Exception:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "Bad Request: Invalid JSON body"}),
        }

    file_name = body.get("fileName")
    file_type = body.get("fileType", "").lower().strip()
    byte_size = body.get("byteSize")

    if not file_name or not file_type or byte_size is None:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "error": "Bad Request: Required fields missing. Must include fileName, fileType, and byteSize."
            }),
        }

    # 3. Guardrail validation: File size and MIME type
    try:
        byte_size = int(byte_size)
    except (ValueError, TypeError):
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "Bad Request: byteSize must be a valid integer"}),
        }

    if byte_size <= 0:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "Bad Request: byteSize must be greater than 0"}),
        }

    if byte_size > MAX_FILE_SIZE_BYTES:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "error": f"Payload Too Large: File exceeds maximum allowed size of 50 MB ({MAX_FILE_SIZE_BYTES} bytes)"
            }),
        }

    # Normalize mime / extension checks
    is_valid_mime = file_type in ALLOWED_MIME_TYPES
    is_valid_ext = file_name.lower().endswith(".csv") or file_name.lower().endswith(".xlsx")
    if not (is_valid_mime or is_valid_ext):
        return {
            "statusCode": 415,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "error": "Unsupported Media Type: Only CSV and XLSX formats are supported."
            }),
        }

    # 4. Generate unique job_id and S3 key
    job_id = f"job_{uuid.uuid4().hex}"
    ext = ".xlsx" if file_name.lower().endswith(".xlsx") or "spreadsheetml" in file_type else ".csv"
    s3_key = f"{tenant_id}/csv/{job_id}{ext}"

    # 5. Generate presigned PUT URL (15-minute expiration = 900 seconds)
    params = {
        "Bucket": INGESTION_BUCKET,
        "Key": s3_key,
        "ContentType": file_type if file_type in ALLOWED_MIME_TYPES else "text/csv",
        "Metadata": {
            "tenant_id": tenant_id,
            "original_filename": file_name,
            "job_id": job_id,
        },
    }

    try:
        presigned_url = s3_client.generate_presigned_url(
            ClientMethod="put_object",
            Params=params,
            ExpiresIn=900,
        )
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": f"Failed to generate presigned URL: {str(e)}"}),
        }

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps({
            "uploadUrl": presigned_url,
            "s3Key": s3_key,
            "jobId": job_id,
            "bucket": INGESTION_BUCKET,
            "expiresIn": 900,
        }),
    }
