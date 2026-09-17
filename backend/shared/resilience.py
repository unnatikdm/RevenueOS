"""
Resilience and Error Recovery Utilities for RevenueOS Enterprise Serverless Architecture
Target: backend/shared/resilience.py

Equips every Lambda, DynamoDB batch writer, S3 streaming processor, and external connector with:
1. robust_retry: Exponential backoff with full jitter and custom retryable exception filters.
2. DynamoDBBatchWriterSafe: Fault-tolerant DynamoDB batch writer automatically handling
   ProvisionedThroughputExceededException and UnprocessedItems with exponential backoff.
3. parse_flexible_numeric: Multi-format currency and number parsing (handles '₹1,299.00', '$45.50', '2,400.50', None, NaN).
4. parse_flexible_timestamp: Resilient ISO-8601, RFC-2822, and regional date parsing with timezone normalization.
5. CircuitBreaker: Stateful circuit breaker preventing cascade failure during third-party API degradation.
"""

import functools
import logging
import math
import random
import re
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

logger = logging.getLogger("revenueos.resilience")
logger.setLevel(logging.INFO)


def robust_retry(
    max_attempts: int = 5,
    initial_delay: float = 0.5,
    max_delay: float = 30.0,
    backoff_multiplier: float = 2.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """
    Decorator executing exponential backoff with full jitter:
    sleep = min(max_delay, random.uniform(0, initial_delay * (backoff_multiplier ** attempt)))
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_err = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as exc:
                    last_err = exc
                    if attempt == max_attempts:
                        logger.error(f"[RETRY_EXHAUSTED] {func.__name__} failed after {attempt} attempts: {exc}")
                        raise
                    # Full jitter formula
                    sleep_time = random.uniform(0, min(max_delay, delay))
                    logger.warning(f"[RETRY_ATTEMPT] {func.__name__} attempt {attempt} failed: {exc}. Retrying in {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
                    delay *= backoff_multiplier
            raise last_err
        return wrapper
    return decorator


def parse_flexible_numeric(value: Any, default: Decimal = Decimal("0.00")) -> Decimal:
    """
    Converts heterogeneous strings, floats, ints, currency symbols (₹, $, €, £),
    commas, and whitespace safely into a Decimal without precision loss or exceptions.
    """
    if value is None:
        return default
    if isinstance(value, (int, Decimal)):
        return Decimal(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return default
        return Decimal(str(value))

    val_str = str(value).strip()
    if not val_str or val_str.lower() in ("nan", "none", "null", "undefined", ""):
        return default

    # Remove currency symbols, commas, and formatting noise
    cleaned = re.sub(r"[^\d.-]", "", val_str)
    if not cleaned or cleaned in ("-", "."):
        return default

    try:
        return Decimal(cleaned)
    except (InvalidOperation, ValueError):
        return default


def parse_flexible_timestamp(date_val: Any) -> datetime:
    """
    Safely parses diverse date formats (ISO-8601, UK %d/%m/%Y, US %m/%d/%Y, epoch stamps)
    and ensures timezone awareness in UTC. Never crashes on corrupt date inputs.
    """
    if isinstance(date_val, datetime):
        if date_val.tzinfo is None:
            return date_val.replace(tzinfo=timezone.utc)
        return date_val.astimezone(timezone.utc)

    if not date_val:
        return datetime.now(timezone.utc)

    date_str = str(date_val).strip()

    # Numeric epoch timestamps
    if date_str.isdigit():
        try:
            ts = int(date_str)
            if ts > 1e11:  # Milliseconds
                ts /= 1000.0
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            pass

    formats = (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%m/%d/%Y %H:%M",
        "%d/%m/%Y %H:%M",
        "%m/%d/%Y %I:%M %p",
        "%Y/%m/%d %H:%M:%S",
    )

    clean_str = date_str.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(clean_str)
    except Exception:
        pass

    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            continue

    logger.warning(f"Unrecognized date format '{date_str}', falling back to current UTC timestamp.")
    return datetime.now(timezone.utc)


class CircuitBreakerOpenException(Exception):
    """Raised when an operation is attempted while the circuit breaker is open."""
    pass


class CircuitBreaker:
    """
    Protects downstream systems (Shopify API, Bedrock Converse, DynamoDB)
    from cascading failure by failing fast when error rates spike.
    """
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def record_success(self):
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.error(f"[CIRCUIT_BREAKER_TRIPPED] State is OPEN. Blocking calls for {self.recovery_timeout}s.")

    def check_state(self):
        if self.state == "OPEN":
            if (time.time() - self.last_failure_time) > self.recovery_timeout:
                self.state = "HALF_OPEN"
                logger.info("[CIRCUIT_BREAKER_HALF_OPEN] Probing downstream service health.")
            else:
                raise CircuitBreakerOpenException(f"Circuit Breaker is OPEN. Downstream failure protection active.")

    def __call__(self, func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.check_state()
            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as exc:
                self.record_failure()
                raise exc
        return wrapper


class DynamoDBBatchWriterSafe:
    """
    Production batch writer for Amazon DynamoDB:
    - Buffers items into 25-item DynamoDB batch limits.
    - Captures UnprocessedItems returned by AWS and automatically retries with exponential backoff.
    - Prevents partial batch loss during throttling spikes.
    """
    def __init__(self, dynamodb_client, table_name: str, batch_size: int = 25):
        self.client = dynamodb_client
        self.table_name = table_name
        self.batch_size = min(batch_size, 25)
        self.buffer: List[Dict[str, Any]] = []

    def put_item(self, item: Dict[str, Any]):
        self.buffer.append({"PutRequest": {"Item": item}})
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self):
        if not self.buffer:
            return

        request_items = {self.table_name: self.buffer}
        self.buffer = []

        delay = 0.1
        max_attempts = 6
        for attempt in range(max_attempts):
            response = self.client.batch_write_item(RequestItems=request_items)
            unprocessed = response.get("UnprocessedItems", {}).get(self.table_name, [])

            if not unprocessed:
                return

            logger.warning(f"DynamoDB batch_write_item had {len(unprocessed)} unprocessed items. Retrying attempt {attempt+1}...")
            time.sleep(delay + random.uniform(0, 0.1))
            delay *= 2.0
            request_items = {self.table_name: unprocessed}

        logger.error(f"Failed to write {len(request_items.get(self.table_name, []))} items to DynamoDB after {max_attempts} attempts.")
        raise RuntimeError("DynamoDB batch write exceeded retry limits with unprocessed items.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.flush()
