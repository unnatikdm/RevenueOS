import json
import unittest
from unittest.mock import MagicMock, patch

from backend.functions.normalization.error_handler import (
    parse_dlq_message,
    record_job_failure,
    lambda_handler,
)


class TestNormalizationErrorHandler(unittest.TestCase):
    def test_parse_dlq_message(self):
        record = {
            "messageId": "msg-dlq-991",
            "attributes": {"ApproximateReceiveCount": "3"},
            "body": json.dumps({
                "tenant_id": "tenant_apex_fashion",
                "job_id": "job_corrupt123",
                "bucket": "revenueos-raw-dev",
                "key": "tenant_apex_fashion/csv/job_corrupt123.csv",
                "error_code": "CORRUPTED_STREAM",
                "error_message": "Invalid UTF-8 sequence encountered in CSV stream",
            }),
        }

        info = parse_dlq_message(record)
        self.assertEqual(info["tenant_id"], "tenant_apex_fashion")
        self.assertEqual(info["job_id"], "job_corrupt123")
        self.assertEqual(info["error_code"], "CORRUPTED_STREAM")
        self.assertEqual(info["receive_count"], 3)

    @patch("backend.functions.normalization.error_handler.table")
    def test_record_job_failure_writes_to_dynamo(self, mock_table):
        failure_info = {
            "tenant_id": "tenant_apex_fashion",
            "job_id": "job_corrupt123",
            "message_id": "msg-dlq-991",
            "bucket": "revenueos-raw-dev",
            "key": "tenant_apex_fashion/csv/job_corrupt123.csv",
            "receive_count": 3,
            "error_code": "CORRUPTED_STREAM",
            "error_message": "Unrecoverable stream corruption",
        }

        record_job_failure(failure_info)

        mock_table.put_item.assert_called_once()
        item = mock_table.put_item.call_args[1]["Item"]
        self.assertEqual(item["PK"], "TENANT#tenant_apex_fashion")
        self.assertEqual(item["SK"], "JOB#job_corrupt123")
        self.assertEqual(item["GSI1-PK"], "TENANT#tenant_apex_fashion#JOBS")
        self.assertEqual(item["status"], "FAILED")
        self.assertEqual(item["error_code"], "CORRUPTED_STREAM")
        self.assertEqual(item["retry_attempts"], 3)

    @patch("backend.functions.normalization.error_handler.record_job_failure")
    def test_lambda_handler_processes_batch(self, mock_record):
        event = {
            "Records": [
                {
                    "messageId": "msg-1",
                    "attributes": {"ApproximateReceiveCount": "3"},
                    "body": json.dumps({"tenant_id": "tenant_apex_fashion", "job_id": "job_1"}),
                },
                {
                    "messageId": "msg-2",
                    "attributes": {"ApproximateReceiveCount": "3"},
                    "body": json.dumps({"tenant_id": "tenant_apex_fashion", "job_id": "job_2"}),
                },
            ]
        }

        res = lambda_handler(event, None)
        self.assertEqual(res["statusCode"], 200)
        self.assertEqual(res["processed_dlq_count"], 2)
        self.assertEqual(mock_record.call_count, 2)


if __name__ == "__main__":
    unittest.main()
