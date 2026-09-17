from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator


ISO_UTC_MILLIS = "%Y-%m-%dT%H:%M:%S.%fZ"


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def format_iso_utc(value: datetime) -> str:
    utc_value = to_utc(value)
    return utc_value.strftime(ISO_UTC_MILLIS)[:-4] + "Z"


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
        validate_assignment=True,
        use_enum_values=True,
    )

    @field_serializer("*", when_used="json")
    def serialize_fields(self, value: Any) -> Any:
        if isinstance(value, datetime):
            return format_iso_utc(value)
        if isinstance(value, Decimal):
            return str(value)
        return value


class Organization(StrictBaseModel):
    tenant_id: str = Field(..., min_length=1)
    org_name: str = Field(..., min_length=1)
    tier: Literal["STANDARD", "PRO", "ENTERPRISE"] = "STANDARD"
    created_at: datetime = Field(default_factory=now_utc)
    settings: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def created_at_utc(cls, value: datetime) -> datetime:
        return to_utc(value)


class User(StrictBaseModel):
    tenant_id: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)
    email: str = Field(..., min_length=3)
    role: Literal["ADMIN", "ANALYST", "VIEWER"] = "ANALYST"
    cognito_sub: Optional[str] = None
    created_at: datetime = Field(default_factory=now_utc)
    last_login: Optional[datetime] = None

    @field_validator("created_at", "last_login")
    @classmethod
    def user_timestamps_utc(cls, value: Optional[datetime]) -> Optional[datetime]:
        return to_utc(value) if value else value


class DataSource(StrictBaseModel):
    tenant_id: str = Field(..., min_length=1)
    conn_id: str = Field(..., min_length=1)
    provider: Literal["SHOPIFY", "S3_CSV", "S3_XLSX"]
    status: Literal["ACTIVE", "PENDING", "INACTIVE", "FAILED"] = "ACTIVE"
    secrets_arn: Optional[str] = None
    last_sync_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=now_utc)

    @field_validator("created_at", "last_sync_at")
    @classmethod
    def datasource_timestamps_utc(cls, value: Optional[datetime]) -> Optional[datetime]:
        return to_utc(value) if value else value


class Product(StrictBaseModel):
    tenant_id: str = Field(..., min_length=1)
    product_id: str = Field(..., min_length=1)
    sku: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    vendor: Optional[str] = None
    price: Decimal = Field(..., ge=Decimal("0.00"))
    cost: Optional[Decimal] = Field(default=Decimal("0.00"), ge=Decimal("0.00"))
    currency: str = Field(default="INR", min_length=3, max_length=3)
    status: Literal["ACTIVE", "ARCHIVED", "DRAFT"] = "ACTIVE"


class ProductVariant(StrictBaseModel):
    tenant_id: str = Field(..., min_length=1)
    variant_id: str = Field(..., min_length=1)
    product_id: str = Field(..., min_length=1)
    sku: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    price: Decimal = Field(..., ge=Decimal("0.00"))
    inventory_quantity: int = Field(default=0, ge=0)


class InventorySnapshot(StrictBaseModel):
    tenant_id: str = Field(..., min_length=1)
    product_id: str = Field(..., min_length=1)
    sku: str = Field(..., min_length=1)
    snapshot_date: date
    available_units: int = Field(..., ge=0)
    reserved_units: int = Field(default=0, ge=0)
    reorder_point: Optional[int] = Field(default=None, ge=0)

    @field_validator("snapshot_date", mode="before")
    @classmethod
    def parse_snapshot_date(cls, value: Any) -> Any:
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        if isinstance(value, str):
            return date.fromisoformat(value)
        return value

    @field_serializer("snapshot_date", when_used="json")
    def serialize_snapshot_date(self, value: date) -> str:
        return value.isoformat()


class OrderItem(StrictBaseModel):
    item_id: str = Field(..., min_length=1)
    product_id: str = Field(..., min_length=1)
    variant_id: Optional[str] = None
    sku: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    quantity: int = Field(..., gt=0)
    unit_price: Decimal = Field(..., ge=Decimal("0.00"))
    total_discount: Decimal = Field(default=Decimal("0.00"), ge=Decimal("0.00"))
    tax_amount: Decimal = Field(default=Decimal("0.00"), ge=Decimal("0.00"))


class Order(StrictBaseModel):
    tenant_id: str = Field(..., min_length=1)
    order_id: str = Field(..., min_length=1)
    source: str = Field(default="SHOPIFY", min_length=1)
    customer_id: Optional[str] = None
    created_at: datetime
    currency: str = Field(default="INR", min_length=3, max_length=3)
    gross_amount: Decimal = Field(..., ge=Decimal("0.00"))
    net_amount: Decimal = Field(..., ge=Decimal("0.00"))
    total_tax: Decimal = Field(default=Decimal("0.00"), ge=Decimal("0.00"))
    total_discounts: Decimal = Field(default=Decimal("0.00"), ge=Decimal("0.00"))
    financial_status: str = Field(default="PAID", min_length=1)
    items: List[OrderItem] = Field(..., min_length=1)

    @field_validator("created_at")
    @classmethod
    def order_created_at_utc(cls, value: datetime) -> datetime:
        return to_utc(value)


class Return(StrictBaseModel):
    tenant_id: str = Field(..., min_length=1)
    return_id: str = Field(..., min_length=1)
    order_id: str = Field(..., min_length=1)
    product_id: str = Field(..., min_length=1)
    sku: str = Field(..., min_length=1)
    returned_at: datetime
    refund_amount: Decimal = Field(..., ge=Decimal("0.00"))
    return_reason: str = Field(..., min_length=1)
    restock_status: Literal["RESTOCKED", "DAMAGED", "SALVAGED", "DISCARDED"] = "RESTOCKED"

    @field_validator("returned_at")
    @classmethod
    def returned_at_utc(cls, value: datetime) -> datetime:
        return to_utc(value)


class LeakEvidence(StrictBaseModel):
    daily_demand_rate: Optional[Decimal] = Field(default=None, ge=Decimal("0.0"))
    demand_sample_size: Optional[int] = Field(default=None, ge=0)
    demand_sample_variance: Optional[Decimal] = Field(default=None, ge=Decimal("0.0"))
    ddr_confidence_interval_low: Optional[Decimal] = None
    ddr_confidence_interval_high: Optional[Decimal] = None
    stockout_days: Optional[int] = Field(default=None, ge=0)
    average_selling_price: Optional[Decimal] = Field(default=None, ge=Decimal("0.0"))
    estimated_missed_units: Optional[int] = Field(default=None, ge=0)
    baseline_return_rate: Optional[Decimal] = Field(default=None, ge=Decimal("0.0"), le=Decimal("1.0"))
    current_return_rate: Optional[Decimal] = Field(default=None, ge=Decimal("0.0"), le=Decimal("1.0"))
    z_score: Optional[Decimal] = None
    p_value_alpha: Optional[Decimal] = None
    percentage_increase: Optional[Decimal] = None
    top_correlated_reason: Optional[str] = None
    reason_prevalence: Optional[Decimal] = Field(default=None, ge=Decimal("0.0"), le=Decimal("1.0"))
    price_elasticity: Optional[Decimal] = None
    baseline_price: Optional[Decimal] = Field(default=None, ge=Decimal("0.0"))
    post_change_price: Optional[Decimal] = Field(default=None, ge=Decimal("0.0"))
    baseline_units: Optional[int] = Field(default=None, ge=0)
    post_change_units: Optional[int] = Field(default=None, ge=0)
    raw_metrics: Dict[str, Any] = Field(default_factory=dict)


class Leak(StrictBaseModel):
    tenant_id: str = Field(..., min_length=1)
    leak_id: str = Field(..., min_length=1)
    type: str = Field(..., min_length=1)
    entity_id: str = Field(..., min_length=1)
    impact_amount: Decimal = Field(..., ge=Decimal("0.00"))
    opp_score: Decimal = Field(..., ge=Decimal("0.0"), le=Decimal("100.0"))
    confidence: Decimal = Field(..., ge=Decimal("0.0"), le=Decimal("1.0"))
    evidence: LeakEvidence
    detected_at: datetime = Field(default_factory=now_utc)
    status: Literal["ACTIVE", "RESOLVED", "DISMISSED", "INSUFFICIENT_DATA"] = "ACTIVE"

    @field_validator("detected_at")
    @classmethod
    def leak_detected_at_utc(cls, value: datetime) -> datetime:
        return to_utc(value)


class Recommendation(StrictBaseModel):
    tenant_id: str = Field(..., min_length=1)
    rec_id: str = Field(..., min_length=1)
    leak_id: str = Field(..., min_length=1)
    priority: Literal["P0", "P1", "P2", "P3"] = "P1"
    root_cause: str = Field(..., min_length=1)
    action_text: str = Field(..., min_length=1)
    expected_recovery: Decimal = Field(..., ge=Decimal("0.00"))
    expected_recovery_pct: Decimal = Field(default=Decimal("0.20"), ge=Decimal("0.0"), le=Decimal("1.0"))
    operational_effort: Literal["LOW", "MEDIUM", "HIGH"] = "MEDIUM"
    contributing_factors: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=now_utc)

    @field_validator("created_at")
    @classmethod
    def recommendation_created_at_utc(cls, value: datetime) -> datetime:
        return to_utc(value)


class AnalysisRun(StrictBaseModel):
    tenant_id: str = Field(..., min_length=1)
    run_id: str = Field(..., min_length=1)
    status: Literal["RUNNING", "SUCCEEDED", "FAILED"] = "RUNNING"
    created_at: datetime = Field(default_factory=now_utc)
    leaks_found: int = Field(default=0, ge=0)
    total_opportunity: Decimal = Field(default=Decimal("0.00"), ge=Decimal("0.00"))
    duration_ms: Optional[int] = Field(default=None, ge=0)
    error_message: Optional[str] = None

    @field_validator("created_at")
    @classmethod
    def analysis_created_at_utc(cls, value: datetime) -> datetime:
        return to_utc(value)


CanonicalOrderItem = OrderItem
CanonicalOrder = Order
CanonicalInventorySnapshot = InventorySnapshot
CanonicalReturn = Return
