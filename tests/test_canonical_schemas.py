import unittest
from datetime import datetime, timezone
from decimal import Decimal
from pydantic import ValidationError

from backend.shared.schemas.canonical import (
    Organization,
    User,
    DataSource,
    Product,
    ProductVariant,
    InventorySnapshot,
    OrderItem,
    Order,
    Return,
    LeakEvidence,
    Leak,
    Recommendation,
    AnalysisRun,
)


class TestCanonicalSchemas(unittest.TestCase):
    def test_organization_model(self):
        org = Organization(tenant_id="tenant_123", org_name="Apex Retail")
        self.assertEqual(org.tenant_id, "tenant_123")
        self.assertEqual(org.tier, "STANDARD")

        with self.assertRaises(ValidationError):
            # Empty tenant_id rejected
            Organization(tenant_id="", org_name="Apex")

    def test_user_model(self):
        user = User(tenant_id="tenant_123", user_id="user_1", email="analyst@apex.com", role="ADMIN")
        self.assertEqual(user.role, "ADMIN")
        self.assertEqual(user.email, "analyst@apex.com")

        with self.assertRaises(ValidationError):
            User(tenant_id="tenant_123", user_id="user_1", email="ab")

    def test_order_and_item_decimal_precision(self):
        now = datetime.now(timezone.utc)
        item = OrderItem(
            item_id="item_1",
            product_id="prod_1",
            sku="SKU-104",
            title="Running Shoes Black",
            quantity=2,
            unit_price=Decimal("1299.99"),
            total_discount=Decimal("50.00"),
            tax_amount=Decimal("224.99"),
        )
        order = Order(
            tenant_id="tenant_123",
            order_id="ord_1001",
            source="SHOPIFY",
            created_at=now,
            gross_amount=Decimal("2599.98"),
            net_amount=Decimal("2549.98"),
            financial_status="PAID",
            items=[item],
        )
        self.assertIsInstance(order.gross_amount, Decimal)
        self.assertEqual(order.gross_amount, Decimal("2599.98"))
        self.assertIsInstance(order.items[0].unit_price, Decimal)
        self.assertEqual(order.items[0].unit_price, Decimal("1299.99"))

        # Immutability check
        with self.assertRaises(ValidationError):
            order.gross_amount = Decimal("3000.00")

    def test_inventory_snapshot_validation(self):
        inv = InventorySnapshot(
            tenant_id="tenant_123",
            product_id="prod_1",
            sku="SKU-104",
            snapshot_date="2026-03-01",
            available_units=150,
            reserved_units=10,
        )
        self.assertEqual(inv.available_units, 150)

        # Negative available_units rejected
        with self.assertRaises(ValidationError):
            InventorySnapshot(
                tenant_id="tenant_123",
                product_id="prod_1",
                sku="SKU-104",
                snapshot_date="2026-03-01",
                available_units=-5,
            )

        # Invalid date pattern rejected
        with self.assertRaises(ValidationError):
            InventorySnapshot(
                tenant_id="tenant_123",
                product_id="prod_1",
                sku="SKU-104",
                snapshot_date="01-03-2026",
                available_units=10,
            )

    def test_return_model(self):
        now = datetime.now(timezone.utc)
        ret = Return(
            tenant_id="tenant_123",
            return_id="ret_55",
            order_id="ord_1001",
            product_id="prod_1",
            sku="SKU-104",
            returned_at=now,
            refund_amount=Decimal("1299.99"),
            return_reason="Wrong Size",
        )
        self.assertEqual(ret.refund_amount, Decimal("1299.99"))
        self.assertEqual(ret.restock_status, "RESTOCKED")

    def test_leak_and_evidence(self):
        evidence = LeakEvidence(
            daily_demand_rate=3.45,
            stockout_days=7,
            average_selling_price=1299.00,
            estimated_missed_units=24,
        )
        leak = Leak(
            tenant_id="tenant_123",
            leak_id="leak_stockout_104",
            type="stockout",
            entity_id="SKU-104",
            impact_amount=Decimal("31176.00"),
            opp_score=Decimal("84.5"),
            confidence=Decimal("0.95"),
            evidence=evidence,
        )
        self.assertEqual(leak.opp_score, Decimal("84.5"))
        self.assertEqual(leak.evidence.stockout_days, 7)

    def test_recommendation_and_analysis_run(self):
        rec = Recommendation(
            tenant_id="tenant_123",
            rec_id="rec_001",
            leak_id="leak_stockout_104",
            priority="P1",
            root_cause="Supplier lead time breach combined with steady demand",
            action_text="Expedite PO #8821 and activate safety stock alerts at 15 units.",
            expected_recovery=Decimal("24000.00"),
            expected_recovery_pct=Decimal("0.77"),
            operational_effort="LOW",
        )
        self.assertEqual(rec.priority, "P1")
        self.assertEqual(rec.expected_recovery, Decimal("24000.00"))

        run = AnalysisRun(
            tenant_id="tenant_123",
            run_id="run_99",
            status="SUCCEEDED",
            leaks_found=5,
            total_opportunity=Decimal("842000.00"),
            duration_ms=4520,
        )
        self.assertEqual(run.leaks_found, 5)
        self.assertEqual(run.total_opportunity, Decimal("842000.00"))


if __name__ == "__main__":
    unittest.main()
