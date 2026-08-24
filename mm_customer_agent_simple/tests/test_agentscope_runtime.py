from __future__ import annotations

import importlib.util
import unittest

from mm_customer_agent_simple.src.agentscope_runtime import (
    AgentScopeCustomerServiceRuntime,
)
from mm_customer_agent_simple.src.config import Settings

try:
    from mm_customer_agent_simple.src.agentscope_application import (
        AgentScopeExecutionContext,
        AgentScopeMultiAgentApplication,
    )
except ModuleNotFoundError:
    AgentScopeExecutionContext = None
    AgentScopeMultiAgentApplication = None


AGENTSCOPE_INSTALLED = importlib.util.find_spec("agentscope") is not None


@unittest.skipUnless(AGENTSCOPE_INSTALLED, "AgentScope is not installed")
class AgentScopeRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.runtime = AgentScopeCustomerServiceRuntime(
            Settings(dashscope_api_key="test-key")
        )

    def test_order_tool_is_registered_with_schema(self):
        schemas = self.runtime.toolkit.get_json_schemas()

        self.assertEqual(schemas[0]["function"]["name"], "query_order")
        self.assertIn(
            "order_id",
            schemas[0]["function"]["parameters"]["properties"],
        )

    def test_toolkit_executes_mock_order_query_without_llm(self):
        result = self.runtime.lookup_order_sync("ORD202608230001")

        self.assertTrue(result["found"])
        self.assertEqual(result["order"]["status"], "shipped")


@unittest.skipUnless(
    AGENTSCOPE_INSTALLED and AgentScopeMultiAgentApplication is not None,
    "AgentScope or project runtime dependencies are not installed",
)
class AgentScopeApplicationTests(unittest.TestCase):
    def setUp(self):
        self.app = AgentScopeMultiAgentApplication(  # type: ignore[misc]
            Settings(dashscope_api_key="test-key")
        )

    def test_main_agent_exposes_only_agent_delegation_tools(self):
        context = AgentScopeExecutionContext()  # type: ignore[misc]
        toolkit = self.app._build_main_toolkit(context, [], [])
        names = {
            item["function"]["name"] for item in toolkit.get_json_schemas()
        }

        self.assertEqual(
            names,
            {"delegate_customer_service", "delegate_manual_qa"},
        )

    def test_public_flow_uses_main_and_customer_agentscope_agents(self):
        imports = self.app.framework._load_agentscope()

        async def invoke(toolkit, name, input_data):
            responses = await toolkit.call_tool_function(
                imports["ToolUseBlock"](
                    type="tool_use",
                    id=f"test-{name}",
                    name=name,
                    input=input_data,
                )
            )
            async for _ in responses:
                pass

        class FakeMessage:
            def __init__(self, text):
                self.text = text

            def get_text_content(self):
                return self.text

        class FakeAgent:
            def __init__(self, name, toolkit):
                self.name = name
                self.toolkit = toolkit

            async def __call__(self, _message):
                if self.name == "main_customer_agent":
                    await invoke(
                        self.toolkit,
                        "delegate_customer_service",
                        {"query": "查询订单 ORD202608230001 的物流"},
                    )
                    return FakeMessage("订单正在运输中。")
                if self.name == "customer_service_agent":
                    await invoke(
                        self.toolkit,
                        "query_order",
                        {"order_id": "ORD202608230001"},
                    )
                    return FakeMessage("订单正在运输中。")
                raise AssertionError(self.name)

        self.app._build_react_agent = (
            lambda name, system_prompt, toolkit: FakeAgent(name, toolkit)
        )
        result = self.app.answer_sync("查询订单 ORD202608230001 的物流")

        self.assertEqual(result["framework"], "agentscope")
        self.assertEqual(result["selected_sub_agent"], "customer_service")
        self.assertEqual(result["route"], "mock_order_query")
        self.assertTrue(result["order_query"]["found"])

    def test_public_flow_uses_manual_agentscope_agent_and_search_tool(self):
        from mm_customer_agent_simple.src.schema import RetrievedChunk

        imports = self.app.framework._load_agentscope()

        async def invoke(toolkit, name, input_data):
            responses = await toolkit.call_tool_function(
                imports["ToolUseBlock"](
                    type="tool_use",
                    id=f"test-{name}",
                    name=name,
                    input=input_data,
                )
            )
            async for _ in responses:
                pass

        class FakeSearch:
            def search(self, query, top_k=None):
                return [
                    RetrievedChunk(
                        chunk_id="manual-1",
                        manual_id="空气净化器手册",
                        section_path="更换滤网",
                        text="关闭电源后打开后盖并更换滤网。",
                        image_ids=["Manual03_10"],
                        score=0.9,
                    )
                ]

        class FakeMessage:
            def __init__(self, text):
                self.text = text

            def get_text_content(self):
                return self.text

        class FakeAgent:
            def __init__(self, name, toolkit):
                self.name = name
                self.toolkit = toolkit

            async def __call__(self, _message):
                if self.name == "main_customer_agent":
                    await invoke(
                        self.toolkit,
                        "delegate_manual_qa",
                        {"query": "空气净化器滤网怎么更换？"},
                    )
                    return FakeMessage("关闭电源后打开后盖并更换滤网。")
                if self.name == "manual_qa_agent":
                    await invoke(
                        self.toolkit,
                        "search_manual",
                        {"search_query": "空气净化器 更换滤网", "top_k": 4},
                    )
                    return FakeMessage("关闭电源后打开后盖并更换滤网。")
                raise AssertionError(self.name)

        self.app._manual_search = FakeSearch()
        self.app._build_react_agent = (
            lambda name, system_prompt, toolkit: FakeAgent(name, toolkit)
        )
        result = self.app.answer_sync("空气净化器滤网怎么更换？")

        self.assertEqual(result["selected_sub_agent"], "manual")
        self.assertEqual(result["route"], "agentscope_manual_rag")
        self.assertEqual(result["image_ids"], ["Manual03_10"])


if __name__ == "__main__":
    unittest.main()
