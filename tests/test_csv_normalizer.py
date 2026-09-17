import io
import json
import unittest
from decimal import Decimal
from unittest.mock import MagicMock, patch

from backend.functions.normalization.csv_normalizer import (
    normalize_row_dict,
    process_csv_file,
    lambda_handler,
)


class TestCsvNormalizer(unittest.TestCase):
    def test_normalize_row_dict_column_mapping(self):
        raw = {
            "order_number": "1001",
            "merchant_id": "tenant_apex",
            "client_id": "cust_99",
            "date": "2026-03-01T10:00:00Z",
            "variant_sku": "SKU-104",
            "line_item_name": "Running Shoes",
            "qty": "2",
            "price": "1499.00",
            "total_price": "2998.00",
            "payment_status": "PAID",
        }
        norm = normalize_row_dict(raw)
        self.assertEqual(norm["order_id"], "1001")
        self.assertEqual(norm["tenant_id"], "tenant_apex")
        self.assertEqual(norm["sku"], "SKU-104")
        self.assertEqual(norm["quantity"], "2")
        self.assertEqual(norm["gross_amount"], "2998.00")

    @patch("backend.functions.normalization.csv_normalizer.table")
    @patch("backend.functions.normalization.csv_normalizer.s3_client")
    def test_process_csv_file_successful_batch_write(self, mock_s3, mock_table):
        csv_content = (
            "order_id,tenant_id,created_at,gross_amount,sku,title,quantity,unit_price\n"
            "ORD-001,tenant_apex,2026-03-01T09:00:00Z,2499.00,SKU-104,Running Shoes,1,2499.00\n"
            "ORD-002,tenant_apex,2026-03-01T10:00:00Z,1899.00,SKU-208,Denim,1,1899.00\n"
        )
        mock_body = MagicMock()
        mock_body.read.return_value = csv_content.encode("utf-8")
        # iterdecode expects an iterable yielding bytes
        mock_body.__iter__.return_value = [csv_content.encode("utf-8")]

        mock_s3.get_object.return_value = {"Body": mock_body}

        # Mock DynamoDB batch writer context manager
        mock_batch = MagicMock()
        mock_table.batch_writer.return_value.__enter__.return_value = mock_batch

        result = process_csv_file("revenueos-raw-dev", "tenant_apex/csv/job_test123.csv")

        self.assertEqual(result["tenant_id"], "tenant_apex")
        self.assertEqual(result["processed_rows"], 2)
        self.assertEqual(result["error_rows_count"], 0)
        self.assertEqual(result["status"], "NORMALIZED")

        # Verify DynamoDB batch write calls
        self.assertEqual(mock_batch.put_item.call_count, 2)
        first_call = mock_batch.put_item.call_args_list[0][1]["Item"]
        self.assertEqual(first_call["PK"], "TENANT#tenant_apex")
        self.assertEqual(first_call["SK"], "ORD#ORD-001")
        self.assertEqual(first_call["GSI1-PK"], "TENANT#tenant_apex#ORD")

    @patch("backend.functions.normalization.csv_normalizer.process_csv_file")
    def test_lambda_handler_sqs_trigger(self, mock_process):
        mock_process.return_value = {"tenant_id": "tenant_123", "status": "NORMALIZED"}

        sqs_event = {
            "Records": [
                {
                    "body": json.dumps({
                        "detail": {
                            "bucket": {"name": "revenueos-raw-dev"},
                            "object": {"key": "tenant_123/csv/job_999.csv"},
                        }
                    })
                }
            ]
        }

        res = lambda_handler(sqs_event, None)
        self.assertEqual(res["statusCode"], 200)
        mock_process.assert_called_once_with("revenueos-raw-dev", "tenant_123/csv/job_999.csv")


if __name__ == "__main__":
    unittest.main()
