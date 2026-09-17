from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Literal

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from backend.shared.schemas.canonical import format_iso_utc

logger = logging.getLogger("revenueos.bedrock_reasoner")
logger.setLevel(logging.INFO)

AWS_REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-3-5-sonnet-20240620-v1:0")
DYNAMODB_TABLE_NAME = os.environ.get("CORE_TABLE_NAME", "RevenueOS_Core_dev")

bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION, config=Config(retries={"max_attempts": 3}))
dynamodb_resource = boto3.resource("dynamodb", region_name=os.environ.get("AWS_REGION", "ap-south-1"))
table = dynamodb_resource.Table(DYNAMODB_TABLE_NAME)


class PlaybookSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    executive_summary: str = Field(..., min_length=1, max_length=800)
    primary_root_cause: str = Field(..., min_length=1, max_length=1200)
    contributing_factors: List[str] = Field(default_factory=list, max_length=12)
    recommended_action: str = Field(..., min_length=1, max_length=1200)
    expected_recovery_pct: Decimal = Field(..., ge=Decimal("0.00"), le=Decimal("1.00"))
    operational_effort: Literal["LOW", "MEDIUM", "HIGH"]


TOOL_NAME = "publish_leak_analysis"
PUBLISH_LEAK_TOOL = {
    "tools": [
        {
            "toolSpec": {
                "name": TOOL_NAME,
                "description": "Publish a validated operational playbook for a verified RevenueOS leak.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "executive_summary": {"type": "string"},
                            "primary_root_cause": {"type": "string"},
                            "contributing_factors": {"type": "array", "items": {"type": "string"}},
                            "recommended_action": {"type": "string"},
                            "expected_recovery_pct": {"type": "number", "minimum": 0, "maximum": 1},
                            "operational_effort": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
                        },
                        "required": [
                            "executive_summary",
                            "primary_root_cause",
                            "contributing_factors",
                            "recommended_action",
                            "expected_recovery_pct",
                            "operational_effort",
                        ],
                    }
                },
            }
        }
    ],
    "toolChoice": {"tool": {"name": TOOL_NAME}},
}
PUBLISH_LEAK_ANALYSIS_SCHEMA = PUBLISH_LEAK_TOOL["tools"][0]["toolSpec"]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def _extract_tool_input(response: Dict[str, Any]) -> Dict[str, Any]:
    for block in response.get("output", {}).get("message", {}).get("content", []) or []:
        tool_use = block.get("toolUse")
        if tool_use and tool_use.get("name") == TOOL_NAME:
            return tool_use.get("input") or {}
    raise ValueError("Bedrock response did not invoke the required publish_leak_analysis tool")


def _fallback_playbook(evidence_package: Dict[str, Any], reason: str) -> Dict[str, Any]:
    evidence = evidence_package.get("evidence", {})
    factors = [f"{key}: {value}" for key, value in sorted(_jsonable(evidence).items())]
    if not factors:
        factors = ["No computed evidence fields were supplied."]
    playbook = PlaybookSchema(
        executive_summary="AI playbook generation is temporarily unavailable; deterministic leak metrics remain unchanged.",
        primary_root_cause=f"Root-cause narrative unavailable because Bedrock validation failed: {reason}.",
        contributing_factors=factors[:12],
        recommended_action="Review the verified RevenueOS metrics and assign the leak to the relevant operations owner for manual remediation.",
        expected_recovery_pct=Decimal("0.00"),
        operational_effort="MEDIUM",
    )
    output = playbook.model_dump(mode="python")
    output["expected_recovery_pct"] = float(output["expected_recovery_pct"])
    return output


def generate_root_cause_playbook(evidence_package: Dict[str, Any]) -> Dict[str, Any]:
    immutable_facts = json.dumps(_jsonable(evidence_package), sort_keys=True)
    system_instruction = (
        "You are RevenueOS' revenue operations analyst. You must call the publish_leak_analysis tool. "
        "Treat every numeric metric, rate, count, currency amount, timestamp, SKU, and loss value in the user "
        "message as immutable computed fact. Do not recalculate, round, rename, inflate, discount, or replace "
        "any numeric value. If the facts do not support a claim, omit that claim."
    )
    user_prompt = f"Verified immutable RevenueOS metrics:\n{immutable_facts}"

    max_tokens = 1024
    try:
        for _ in range(2):
            response = bedrock_client.converse(
                modelId=BEDROCK_MODEL_ID,
                messages=[{"role": "user", "content": [{"text": user_prompt}]}],
                system=[{"text": system_instruction}],
                toolConfig=PUBLISH_LEAK_TOOL,
                inferenceConfig={"temperature": 0, "maxTokens": max_tokens},
            )
            if response.get("stopReason") == "max_tokens":
                max_tokens *= 2
                continue
            raw_tool = _extract_tool_input(response)
            output = PlaybookSchema.model_validate(raw_tool).model_dump(mode="python")
            output["expected_recovery_pct"] = float(output["expected_recovery_pct"])
            return output
        return _fallback_playbook(evidence_package, "Bedrock response reached max_tokens twice")
    except (ClientError, ValidationError, ValueError, Exception) as exc:
        logger.warning("Bedrock playbook generation failed: %s", exc)
        return _fallback_playbook(evidence_package, str(exc))


def persist_recommendation(
    tenant_id: str,
    leak_id: str,
    evidence_package: Dict[str, Any],
    playbook: Dict[str, Any],
) -> Dict[str, Any]:
    validated = PlaybookSchema.model_validate(playbook)
    now_iso = format_iso_utc(datetime.now(timezone.utc))
    impact = Decimal(str(evidence_package.get("impact_amount", "0.00")))
    expected_recovery = (impact * validated.expected_recovery_pct).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    item = {
        "PK": f"TENANT#{tenant_id}",
        "SK": f"REC#{leak_id}",
        "GSI1-PK": f"TENANT#{tenant_id}#REC",
        "GSI1-SK": f"P1#{now_iso}",
        "entity_type": "RECOMMENDATION",
        "rec_id": f"REC-{leak_id}",
        "leak_id": leak_id,
        "tenant_id": tenant_id,
        "root_cause": validated.primary_root_cause,
        "action_text": validated.recommended_action,
        "executive_summary": validated.executive_summary,
        "expected_recovery": str(expected_recovery),
        "expected_recovery_pct": str(validated.expected_recovery_pct),
        "operational_effort": validated.operational_effort,
        "contributing_factors": validated.contributing_factors,
        "created_at": now_iso,
    }
    table.put_item(Item=item)
    return item


def _tenant_from_claims(event: Dict[str, Any]) -> str:
    claims = event.get("requestContext", {}).get("authorizer", {}).get("jwt", {}).get("claims", {})
    tenant_id = claims.get("custom:tenant_id")
    if not tenant_id:
        raise PermissionError("Missing verified Cognito custom:tenant_id claim")
    return tenant_id


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    try:
        tenant_id = _tenant_from_claims(event) if "requestContext" in event else event["tenant_id"]
        body = json.loads(event.get("body") or "{}") if isinstance(event.get("body"), str) else event
        if body.get("tenant_id") and body.get("tenant_id") != tenant_id:
            return {"statusCode": 403, "body": json.dumps({"error": "tenant_id override is forbidden"})}
        leak_id = body["leak_id"]
        evidence_package = body["evidence_package"]
        playbook = generate_root_cause_playbook(evidence_package)
        persisted = persist_recommendation(tenant_id, leak_id, evidence_package, playbook)
        return {"statusCode": 200, "body": json.dumps({"recommendation": persisted, "playbook": playbook}, default=str)}
    except PermissionError as exc:
        return {"statusCode": 403, "body": json.dumps({"error": str(exc)})}
    except KeyError as exc:
        return {"statusCode": 400, "body": json.dumps({"error": f"Missing required field: {exc}"})}
