import time
import unittest
from datetime import datetime, timezone
from decimal import Decimal

from backend.shared.resilience import (
    parse_flexible_numeric,
    parse_flexible_timestamp,
    robust_retry,
    CircuitBreaker,
    CircuitBreakerOpenException,
)


class TestResilienceModule(unittest.TestCase):
    def test_parse_flexible_numeric(self):
        self.assertEqual(parse_flexible_numeric("₹1,299.50"), Decimal("1299.50"))
        self.assertEqual(parse_flexible_numeric("$45.00"), Decimal("45.00"))
        self.assertEqual(parse_flexible_numeric("2,450,100.99"), Decimal("2450100.99"))
        self.assertEqual(parse_flexible_numeric("  100  "), Decimal("100"))
        self.assertEqual(parse_flexible_numeric(None), Decimal("0.00"))
        self.assertEqual(parse_flexible_numeric("NaN"), Decimal("0.00"))
        self.assertEqual(parse_flexible_numeric("corrupt_chars"), Decimal("0.00"))

    def test_parse_flexible_timestamp(self):
        dt1 = parse_flexible_timestamp("2026-03-15T12:00:00Z")
        self.assertEqual(dt1.year, 2026)
        self.assertEqual(dt1.tzinfo, timezone.utc)

        dt2 = parse_flexible_timestamp("12/1/2010 8:26")
        self.assertEqual(dt2.year, 2010)

        dt3 = parse_flexible_timestamp("2026-03-15")
        self.assertEqual(dt3.year, 2026)

        dt_fallback = parse_flexible_timestamp("invalid_date_value")
        self.assertIsNotNone(dt_fallback)
        self.assertEqual(dt_fallback.tzinfo, timezone.utc)

    def test_robust_retry_recovers(self):
        attempts = 0

        @robust_retry(max_attempts=3, initial_delay=0.01)
        def unstable_operation():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ConnectionError("Temporary socket timeout")
            return "SUCCESS"

        result = unstable_operation()
        self.assertEqual(result, "SUCCESS")
        self.assertEqual(attempts, 3)

    def test_circuit_breaker_trips(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

        @cb
        def failing_api():
            raise RuntimeError("Downstream API 500 error")

        # First failure
        with self.assertRaises(RuntimeError):
            failing_api()
        self.assertEqual(cb.state, "CLOSED")

        # Second failure trips breaker
        with self.assertRaises(RuntimeError):
            failing_api()
        self.assertEqual(cb.state, "OPEN")

        # Next call blocked immediately by circuit breaker
        with self.assertRaises(CircuitBreakerOpenException):
            failing_api()

        # After recovery timeout, circuit enters HALF_OPEN
        time.sleep(0.15)
        self.assertEqual(cb.state, "OPEN")
        cb.check_state()
        self.assertEqual(cb.state, "HALF_OPEN")


if __name__ == "__main__":
    unittest.main()
