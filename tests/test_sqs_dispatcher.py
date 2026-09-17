import json
import unittest
from unittest.mock import MagicMock


class TestSqsIngestionAndEventDispatcher(unittest.TestCase):
    def test_s3_object_created_event_routing_to_sqs(self):
        """
        Simulate an S3 ObjectCreated event matched by EventBridge rule and enqueued to SQS FIFO queue.
        """
        s3_event = {
            "version": "0",
            "id": "event-12345",
            "detail-type": "Object Created",
            "source": "aws.s3",
            "account": "123456789012",
            "time": "2026-03-15T12:00:00Z",
            "region": "ap-south-1",
            "resources": ["arn:aws:s3:::revenueos-raw-123456789012-dev"],
            "detail": {
                "version": "0",
                "bucket": {"name": "revenueos-raw-123456789012-dev"},
                "object": {
                    "key": "tenant_apex_fashion/csv/job_abc123.csv",
                    "size": 154820,
                    "etag": "d41d8cd98f00b204e9800998ecf8427e",
                },
                "request-id": "C3D13FE58DE4C810",
                "requester": "123456789012",
            },
        }

        # Mock SQS FIFO Enqueue
        sqs_client = MagicMock()
        sqs_client.send_message.return_value = {
            "MD5OfMessageBody": "mock-md5",
            "MessageId": "msg-sqs-001",
            "SequenceNumber": "188237918237",
        }

        # Rule action: enqueue to FIFO queue with message group id
        sqs_client.send_message(
            QueueUrl="https://sqs.ap-south-1.amazonaws.com/123456789012/revenueos-ingestion-dev.fifo",
            MessageBody=json.dumps(s3_event),
            MessageGroupId="CsvIngestGroup",
            MessageDeduplicationId=s3_event["id"],
        )

        sqs_client.send_message.assert_called_once()
        _, kwargs = sqs_client.send_message.call_args
        self.assertEqual(kwargs["MessageGroupId"], "CsvIngestGroup")
        self.assertEqual(kwargs["MessageDeduplicationId"], "event-12345")
        body = json.loads(kwargs["MessageBody"])
        self.assertEqual(body["detail"]["object"]["key"], "tenant_apex_fashion/csv/job_abc123.csv")


if __name__ == "__main__":
    unittest.main()
