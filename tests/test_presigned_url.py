
import json
import unittest
from unittest.mock import MagicMock, patch

from backend.functions.ingestion.presigned_url import lambda_handler, MAX_FILE_SIZE_BYTES


class TestPresignedUrlLambda(unittest.TestCase):
    def setUp(self):
        self.valid_event = {
            "requestContext": {
                "authorizer": {
                    "jwt": {
                        "claims": {
                            "custom:tenant_id": "tenant_apex_fashion",
                            "sub": "user_12345",
                            "email": "operations@apexfashion.com",
                        }
                    }
                }
            },
            "body": json.dumps({
                "fileName": "orders_q1_export.csv",
                "fileType": "text/csv",
                "byteSize": 1024 * 1024,  # 1 MB
            }),
        }

    @patch("backend.functions.ingestion.presigned_url.s3_client")
    def test_successful_presigned_url_generation(self, mock_s3):
        mock_s3.generate_presigned_url.return_value = "https://revenueos-raw-dev.s3.amazonaws.com/tenant_apex_fashion/csv/job_abc123.csv?signature=mock"

        res = lambda_handler(self.valid_event, None)
        self.assertEqual(res["statusCode"], 200)

        data = json.loads(res["body"])
        self.assertIn("uploadUrl", data)
        self.assertIn("jobId", data)
        self.assertTrue(data["s3Key"].startswith("tenant_apex_fashion/csv/"))
        self.assertTrue(data["s3Key"].endswith(".csv"))

        # Verify S3 call arguments
        mock_s3.generate_presigned_url.assert_called_once()
        args, kwargs = mock_s3.generate_presigned_url.call_args
        self.assertEqual(kwargs["ClientMethod"], "put_object")
        self.assertEqual(kwargs["ExpiresIn"], 900)
        self.assertEqual(kwargs["Params"]["Metadata"]["tenant_id"], "tenant_apex_fashion")

    def test_missing_tenant_id_rejection(self):
        event = {
            "requestContext": {"authorizer": {"jwt": {"claims": {}}}},
            "body": json.dumps({"fileName": "data.csv", "fileType": "text/csv", "byteSize": 100}),
        }
        res = lambda_handler(event, None)
        self.assertEqual(res["statusCode"], 401)
        self.assertIn("Unauthorized", res["body"])

    def test_file_size_exceeding_limit_rejection(self):
        event = {
            "requestContext": {
                "authorizer": {"jwt": {"claims": {"custom:tenant_id": "tenant_123"}}}
            },
            "body": json.dumps({
                "fileName": "massive_file.csv",
                "fileType": "text/csv",
                "byteSize": MAX_FILE_SIZE_BYTES + 1,
            }),
        }
        res = lambda_handler(event, None)
        self.assertEqual(res["statusCode"], 400)
        self.assertIn("Payload Too Large", res["body"])

    def test_unsupported_mime_type_rejection(self):
        event = {
            "requestContext": {
                "authorizer": {"jwt": {"claims": {"custom:tenant_id": "tenant_123"}}}
            },
            "body": json.dumps({
                "fileName": "malicious_script.exe",
                "fileType": "application/x-msdownload",
                "byteSize": 5000,
            }),
        }
        res = lambda_handler(event, None)
        self.assertEqual(res["statusCode"], 415)
        self.assertIn("Unsupported Media Type", res["body"])

    def test_empty_or_invalid_body_rejection(self):
        event = {
            "requestContext": {
                "authorizer": {"jwt": {"claims": {"custom:tenant_id": "tenant_123"}}}
            },
            "body": "{invalid-json",
        }
        res = lambda_handler(event, None)
        self.assertEqual(res["statusCode"], 400)


if __name__ == "__main__":
    unittest.main()
