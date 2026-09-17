import json
import unittest
from unittest.mock import MagicMock, patch

from backend.functions.connections.shopify_connector import (
    execute_shopify_query,
    sync_shopify_data,
    lambda_handler,
)


class TestShopifyConnector(unittest.TestCase):
    @patch("backend.functions.connections.shopify_connector.time.sleep")
    @patch("backend.functions.connections.shopify_connector.requests.post")
    def test_rate_limit_throttled_recovery(self, mock_post, mock_sleep):
        """
        Tests that an HTTP 200 response with 'THROTTLED' error triggers exponential backoff
        and subsequently succeeds on the retry without data loss.
        """
        throttled_response = MagicMock()
        throttled_response.status_code = 200
        throttled_response.json.return_value = {
            "errors": [{"message": "Throttled: API rate limit exceeded. Please retry."}]
        }

        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {
            "data": {"orders": {"edges": [], "pageInfo": {"hasNextPage": False}}},
            "extensions": {
                "cost": {
                    "requestedQueryCost": 50,
                    "actualQueryCost": 20,
                    "throttleStatus": {
                        "maximumAvailable": 1000,
                        "currentlyAvailable": 800,
                        "restoreRate": 50,
                    },
                }
            },
        }

        mock_post.side_effect = [throttled_response, success_response]

        payload = execute_shopify_query(
            "test-store.myshopify.com",
            "shpat_mock_token_123",
            "query { orders { edges { node { id } } } }",
            initial_backoff=0.01,
        )

        self.assertEqual(mock_post.call_count, 2)
        mock_sleep.assert_called()
        self.assertIn("data", payload)
        self.assertIn("orders", payload["data"])

    @patch("backend.functions.connections.shopify_connector.table")
    @patch("backend.functions.connections.shopify_connector.execute_shopify_query")
    def test_sync_shopify_data_transforms_to_dynamo(self, mock_query, mock_table):
        """
        Tests pagination and canonical data extraction into DynamoDB batch writer.
        """
        mock_query.return_value = {
            "data": {
                "orders": {
                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                    "edges": [
                        {
                            "node": {
                                "id": "gid://shopify/Order/89102",
                                "name": "#1088",
                                "createdAt": "2026-03-01T12:00:00Z",
                                "currencyCode": "INR",
                                "displayFinancialStatus": "PAID",
                                "currentTotalPriceSet": {"shopMoney": {"amount": "2499.00"}},
                                "totalTaxSet": {"shopMoney": {"amount": "381.20"}},
                                "totalDiscountsSet": {"shopMoney": {"amount": "0.00"}},
                                "customer": {"id": "gid://shopify/Customer/991"},
                                "lineItems": {
                                    "edges": [
                                        {
                                            "node": {
                                                "id": "gid://shopify/LineItem/5001",
                                                "title": "Air Running Shoes",
                                                "sku": "SKU-104",
                                                "quantity": 1,
                                                "originalUnitPriceSet": {"shopMoney": {"amount": "2499.00"}},
                                            }
                                        }
                                    ]
                                },
                            }
                        }
                    ],
                }
            }
        }

        mock_batch = MagicMock()
        mock_table.batch_writer.return_value.__enter__.return_value = mock_batch

        result = sync_shopify_data("tenant_apex_fashion", "test-shop.myshopify.com", "token_123")
        self.assertEqual(result["synced_orders"], 1)
        self.assertEqual(result["synced_items"], 1)
        self.assertEqual(result["status"], "SYNCED")

        # Verify DynamoDB item put
        mock_batch.put_item.assert_called_once()
        item = mock_batch.put_item.call_args[1]["Item"]
        self.assertEqual(item["PK"], "TENANT#tenant_apex_fashion")
        self.assertEqual(item["SK"], "ORD#89102")
        self.assertEqual(item["order_id"], "89102")


if __name__ == "__main__":
    unittest.main()
