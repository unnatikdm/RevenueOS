import unittest
import time
import jwt
from unittest.mock import MagicMock


class MockJwtAuthorizer:
    """
    Simulates the AWS API Gateway HTTP API JWT Authorizer mapped to Cognito User Pool
    with issuer validation, audience verification, and tenant claim enforcement.
    """
    def __init__(self, issuer: str, audience: str):
        self.issuer = issuer
        self.audience = audience

    def authorize(self, auth_header: str | None) -> dict:
        if not auth_header or not auth_header.startswith("Bearer "):
            raise PermissionError("HTTP 401: Unauthorized - Missing or invalid Bearer token format")

        token = auth_header.split(" ")[1]
        try:
            # Decode and verify issuer and audience
            payload = jwt.decode(
                token,
                options={"verify_signature": False, "verify_aud": True, "verify_iss": True},
                audience=self.audience,
                issuer=self.issuer,
            )
            # Enforce expiration
            if payload.get("exp", 0) < time.time():
                raise PermissionError("HTTP 401: Unauthorized - Token expired")

            # Extract mandatory tenant_id claim
            tenant_id = payload.get("custom:tenant_id")
            if not tenant_id:
                raise PermissionError("HTTP 403: Forbidden - Missing custom:tenant_id in token")

            return payload
        except Exception as e:
            raise PermissionError(f"HTTP 401: Unauthorized - {str(e)}")


class TestCognitoJwtAuthorization(unittest.TestCase):
    def setUp(self):
        self.issuer = "https://cognito-idp.ap-south-1.amazonaws.com/ap-south-1_MockPoolId"
        self.client_id = "test-app-client-12345"
        self.authorizer = MockJwtAuthorizer(issuer=self.issuer, audience=self.client_id)

    def test_valid_cognito_jwt(self):
        payload = {
            "iss": self.issuer,
            "aud": self.client_id,
            "sub": "user-uuid-101",
            "custom:tenant_id": "tenant_apex_retail",
            "email": "cfo@apexretail.com",
            "exp": time.time() + 3600,
        }
        token = jwt.encode(payload, "secret-key", algorithm="HS256")
        res = self.authorizer.authorize(f"Bearer {token}")
        self.assertEqual(res["custom:tenant_id"], "tenant_apex_retail")
        self.assertEqual(res["sub"], "user-uuid-101")

    def test_missing_or_malformed_auth_header(self):
        with self.assertRaises(PermissionError) as ctx:
            self.authorizer.authorize(None)
        self.assertIn("HTTP 401", str(ctx.exception))

        with self.assertRaises(PermissionError) as ctx:
            self.authorizer.authorize("InvalidToken")
        self.assertIn("HTTP 401", str(ctx.exception))

    def test_invalid_issuer(self):
        payload = {
            "iss": "https://attacker.example.com",
            "aud": self.client_id,
            "custom:tenant_id": "tenant_apex_retail",
            "exp": time.time() + 3600,
        }
        token = jwt.encode(payload, "secret-key", algorithm="HS256")
        with self.assertRaises(PermissionError) as ctx:
            self.authorizer.authorize(f"Bearer {token}")
        self.assertIn("HTTP 401", str(ctx.exception))

    def test_invalid_audience(self):
        payload = {
            "iss": self.issuer,
            "aud": "wrong-client-id",
            "custom:tenant_id": "tenant_apex_retail",
            "exp": time.time() + 3600,
        }
        token = jwt.encode(payload, "secret-key", algorithm="HS256")
        with self.assertRaises(PermissionError) as ctx:
            self.authorizer.authorize(f"Bearer {token}")
        self.assertIn("HTTP 401", str(ctx.exception))

    def test_missing_tenant_id_rejection(self):
        payload = {
            "iss": self.issuer,
            "aud": self.client_id,
            "sub": "user-uuid-101",
            "exp": time.time() + 3600,
        }
        token = jwt.encode(payload, "secret-key", algorithm="HS256")
        with self.assertRaises(PermissionError) as ctx:
            self.authorizer.authorize(f"Bearer {token}")
        self.assertIn("Missing custom:tenant_id", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
