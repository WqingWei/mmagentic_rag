from __future__ import annotations

import unittest

from mm_customer_agent_simple.src.order_tool import MockOrderQueryTool


class MockOrderQueryToolTests(unittest.TestCase):
    def setUp(self):
        self.tool = MockOrderQueryTool()

    def test_extracts_and_normalizes_order_id(self):
        self.assertEqual(
            self.tool.extract_order_id("请查一下 ord202608230001 的物流"),
            "ORD202608230001",
        )

    def test_known_order_contains_structured_logistics(self):
        result = self.tool.query("ORD202608230001")

        self.assertTrue(result["found"])
        self.assertEqual(result["order"]["status"], "shipped")
        self.assertEqual(result["order"]["carrier"], "顺丰速运")

    def test_unknown_order_does_not_fabricate_data(self):
        result = self.tool.query("ORD999999999999")

        self.assertFalse(result["found"])
        self.assertNotIn("order", result)

    def test_order_intent_without_id_requests_id(self):
        result = self.tool.query_from_text("我的订单到哪里了？")

        self.assertIsNotNone(result)
        self.assertTrue(result["missing_order_id"])
if __name__ == "__main__":
    unittest.main()
