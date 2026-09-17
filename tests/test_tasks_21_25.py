import json
import unittest
from unittest.mock import MagicMock, patch

from backend.functions.dashboard.api_handlers import (
    handle_get_dashboard,
    handle_get_leaks,
    handle_get_leak_detail,
    handle_start_analysis,
    lambda_handler,
)


class TestTasks21To25(unittest.TestCase):
    def test_step_functions_asl_definition_validity(self):
        """Verifies Step Functions ASL JSON structure matches state machine specification."""
        import os
        asl_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "infrastructure",
            "statemachine",
            "leak_orchestrator.asl.json",
        )
        self.assertTrue(os.path.exists(asl_path))
        with open(asl_path, "r") as f:
            asl = json.load(f)
        self.assertEqual(asl["StartAt"], "NormalizeBatch")
        self.assertIn("DetectAllLeaksParallel", asl["States"])
        self.assertIn("GeneratePlaybooksMap", asl["States"])
        self.assertEqual(asl["States"]["DetectAllLeaksParallel"]["Type"], "Parallel")
        self.assertEqual(len(asl["States"]["DetectAllLeaksParallel"]["Branches"]), 5)

    def test_handle_get_dashboard_metrics(self):
        dash = handle_get_dashboard("tenant_apex_fashion")
        self.assertEqual(dash["tenant_id"], "tenant_apex_fashion")
        self.assertIn("kpi_ribbon", dash)
        self.assertIn("category_breakdown", dash)
        self.assertGreater(dash["kpi_ribbon"]["total_recoverable_revenue"], 100000.0)
        self.assertEqual(dash["kpi_ribbon"]["currency"], "INR")

    def test_handle_get_leaks_returns_sorted(self):
        leaks = handle_get_leaks("tenant_apex_fashion")
        self.assertGreaterEqual(len(leaks), 3)
        self.assertEqual(leaks[0]["leak_type"], "stockout")
        self.assertGreaterEqual(leaks[0]["opp_score"], leaks[1]["opp_score"])

    def test_handle_get_leak_detail_contains_playbook(self):
        detail = handle_get_leak_detail("tenant_apex_fashion", "LEAK-STOCKOUT-104")
        self.assertIn("leak", detail)
        self.assertIn("recommendation", detail)
        self.assertIn("action_text", detail["recommendation"])
        self.assertEqual(detail["recommendation"]["priority"], "P1")

    @patch("backend.functions.dashboard.api_handlers.sfn_client")
    def test_start_analysis_triggers_state_machine(self, mock_sfn):
        mock_sfn.start_execution.return_value = {"executionArn": "arn:aws:states:mock:execution"}
        res = handle_start_analysis("tenant_apex_fashion")
        self.assertEqual(res["status"], "RUNNING")
        self.assertIn("job_id", res)
        mock_sfn.start_execution.assert_called_once()

    def test_api_gateway_route_dispatch(self):
        event = {
            "requestContext": {
                "http": {"method": "GET"},
                "authorizer": {"jwt": {"claims": {"custom:tenant_id": "tenant_apex_fashion"}}},
            },
            "rawPath": "/dashboard",
        }
        res = lambda_handler(event, None)
        self.assertEqual(res["statusCode"], 200)
        body = json.loads(res["body"])
        self.assertEqual(body["tenant_id"], "tenant_apex_fashion")


if __name__ == "__main__":
    unittest.main()
