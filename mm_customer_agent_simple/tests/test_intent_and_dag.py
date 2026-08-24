from __future__ import annotations

import importlib.util
import json
import unittest

from mm_customer_agent_simple.src.dag_planner import (
    DAGNode,
    ExecutionPlan,
    build_simple_plan,
    plan_from_json,
)
from mm_customer_agent_simple.src.intent_model import (
    IntentPrediction,
    IntentType,
    RuleIntentClassifier,
)


class IntentClassifierTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.classifier = RuleIntentClassifier()

    async def test_four_intent_classes(self):
        cases = {
            "空气净化器滤网怎么更换？": IntentType.KNOWLEDGE,
            "查询订单 ORD202608230001 的物流": IntentType.CUSTOMER_SERVICE,
            "帮我看看": IntentType.CLARIFICATION,
            "查一下订单物流，同时告诉我净化器滤网怎么换": IntentType.MIXED,
        }
        for query, expected in cases.items():
            with self.subTest(query=query):
                prediction = await self.classifier.classify(query)
                self.assertEqual(prediction.intent, expected)

    async def test_multiturn_image_and_complex_requests_require_planner(self):
        history_prediction = await self.classifier.classify(
            "这个怎么处理？", history=[{"role": "user", "content": "净化器报错"}]
        )
        image_prediction = await self.classifier.classify("这是什么？", has_images=True)
        complex_prediction = await self.classifier.classify(
            "查询物流，同时告诉我如何更换滤网"
        )
        self.assertTrue(history_prediction.needs_planner)
        self.assertTrue(image_prediction.needs_planner)
        self.assertTrue(complex_prediction.needs_planner)


class DAGPlannerTests(unittest.TestCase):
    def test_plan_is_topologically_sorted(self):
        plan = ExecutionPlan(
            intent=IntentType.MIXED,
            reason="image before knowledge",
            source="test",
            nodes=[
                DAGNode("knowledge_1", "knowledge", "处理错误", ["image_1"]),
                DAGNode("customer_1", "customer_service", "查询订单"),
                DAGNode("image_1", "image_understanding", "识别图片"),
            ],
        )
        order = [node.node_id for node in plan.topological_nodes()]
        self.assertLess(order.index("image_1"), order.index("knowledge_1"))

    def test_cycle_is_rejected(self):
        value = {
            "intent": "mixed",
            "nodes": [
                {
                    "node_id": "a",
                    "node_type": "knowledge",
                    "query": "a",
                    "depends_on": ["b"],
                },
                {
                    "node_id": "b",
                    "node_type": "customer_service",
                    "query": "b",
                    "depends_on": ["a"],
                },
            ],
        }
        with self.assertRaisesRegex(ValueError, "cycle"):
            plan_from_json(value, IntentType.MIXED)

    def test_mixed_intent_cannot_bypass_planner(self):
        prediction = IntentPrediction(
            intent=IntentType.MIXED,
            confidence=0.99,
            reason="mixed",
            needs_planner=True,
        )
        with self.assertRaisesRegex(ValueError, "Planner Agent"):
            build_simple_plan(prediction, "混合问题")


AGENTSCOPE_INSTALLED = importlib.util.find_spec("agentscope") is not None


@unittest.skipUnless(AGENTSCOPE_INSTALLED, "AgentScope is not installed")
class MixedPlannerApplicationTests(unittest.TestCase):
    def test_image_request_is_inspected_before_dag_submission(self):
        from mm_customer_agent_simple.src.agentscope_application import (
            AgentScopeExecutionContext,
            AgentScopeMultiAgentApplication,
        )
        from mm_customer_agent_simple.src.config import Settings

        app = AgentScopeMultiAgentApplication(
            Settings(dashscope_api_key="test-key"),
            intent_classifier=RuleIntentClassifier(),
        )
        imports = app.framework._load_agentscope()

        class FakeImageTool:
            calls = 0

            def observe(self, image_paths, image_base64, user_question):
                self.calls += 1
                return {
                    "product_type": "空气净化器",
                    "visible_error_code": "E1",
                    "likely_intent": "knowledge",
                }

        async def invoke(toolkit, name, input_data):
            responses = await toolkit.call_tool_function(
                imports["ToolUseBlock"](
                    type="tool_use", id=f"test-{name}", name=name, input=input_data
                )
            )
            async for _ in responses:
                pass

        class FakeMessage:
            def get_text_content(self):
                return "已提交"

        class FakePlanner:
            def __init__(self, toolkit):
                self.toolkit = toolkit

            async def __call__(self, _message):
                await invoke(self.toolkit, "inspect_request_images", {})
                plan = {
                    "intent": "knowledge",
                    "reason": "图片中是产品错误码",
                    "nodes": [
                        {
                            "node_id": "image_1",
                            "node_type": "image_understanding",
                            "query": "识别图片",
                            "depends_on": [],
                        },
                        {
                            "node_id": "knowledge_1",
                            "node_type": "knowledge",
                            "query": "排查 E1",
                            "depends_on": ["image_1"],
                        },
                    ],
                }
                await invoke(
                    self.toolkit, "submit_plan", {"plan_json": json.dumps(plan)}
                )
                return FakeMessage()

        fake_image_tool = FakeImageTool()
        app.image_tool = fake_image_tool
        app._build_react_agent = lambda name, system_prompt, toolkit: FakePlanner(
            toolkit
        )
        prediction = IntentPrediction(
            intent=IntentType.CLARIFICATION,
            confidence=0.7,
            reason="需要结合图片",
            needs_planner=True,
            source="test",
        )
        context = AgentScopeExecutionContext()
        plan = app.framework._run_sync(
            app._create_execution_plan(
                question="这个错误怎么办？",
                history=[],
                has_images=True,
                image_paths=[],
                image_base64=["fake"],
                prediction=prediction,
                context=context,
            )
        )

        self.assertEqual(fake_image_tool.calls, 1)
        self.assertEqual(context.image_observation["visible_error_code"], "E1")
        self.assertEqual(plan.nodes[1].depends_on, ["image_1"])

    def test_planner_routes_mixed_request_and_summary_merges_results(self):
        from mm_customer_agent_simple.src.agentscope_application import (
            AgentScopeMultiAgentApplication,
        )
        from mm_customer_agent_simple.src.config import Settings
        from mm_customer_agent_simple.src.schema import RetrievedChunk

        class MixedClassifier:
            async def classify(self, query, history=None, has_images=False):
                return IntentPrediction(
                    intent=IntentType.MIXED,
                    confidence=0.96,
                    reason="订单查询与产品操作并存",
                    secondary_intents=[
                        IntentType.CUSTOMER_SERVICE,
                        IntentType.KNOWLEDGE,
                    ],
                    needs_planner=True,
                    source="test",
                )

        app = AgentScopeMultiAgentApplication(
            Settings(dashscope_api_key="test-key"), intent_classifier=MixedClassifier()
        )
        imports = app.framework._load_agentscope()

        async def invoke(toolkit, name, input_data):
            responses = await toolkit.call_tool_function(
                imports["ToolUseBlock"](
                    type="tool_use", id=f"test-{name}", name=name, input=input_data
                )
            )
            async for _ in responses:
                pass

        class FakeSearch:
            def search(self, query, top_k=None):
                return [
                    RetrievedChunk(
                        chunk_id="m-1",
                        manual_id="空气净化器手册",
                        section_path="滤网",
                        text="断电后打开后盖更换滤网。",
                        image_ids=[],
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
                if self.name == "planner_agent":
                    plan = {
                        "intent": "mixed",
                        "reason": "两个独立任务可并行",
                        "nodes": [
                            {
                                "node_id": "customer_1",
                                "node_type": "customer_service",
                                "query": "查询订单 ORD202608230001 的物流",
                                "depends_on": [],
                            },
                            {
                                "node_id": "knowledge_1",
                                "node_type": "knowledge",
                                "query": "空气净化器滤网怎么换",
                                "depends_on": [],
                            },
                        ],
                    }
                    await invoke(
                        self.toolkit, "submit_plan", {"plan_json": json.dumps(plan)}
                    )
                    return FakeMessage("已提交")
                if self.name == "customer_service_agent":
                    await invoke(
                        self.toolkit,
                        "query_order",
                        {"order_id": "ORD202608230001"},
                    )
                    return FakeMessage("订单运输中。")
                if self.name == "manual_qa_agent":
                    await invoke(
                        self.toolkit,
                        "search_manual",
                        {"search_query": "空气净化器滤网", "top_k": 4},
                    )
                    return FakeMessage("断电后打开后盖更换滤网。")
                if self.name == "summary_agent":
                    return FakeMessage("订单运输中；断电后打开后盖更换滤网。")
                raise AssertionError(self.name)

        app._manual_search = FakeSearch()
        app._build_react_agent = lambda name, system_prompt, toolkit: FakeAgent(
            name, toolkit
        )
        result = app.answer_sync(
            "查询订单 ORD202608230001，同时告诉我空气净化器滤网怎么换"
        )

        self.assertEqual(result["execution_plan"]["source"], "planner_agent")
        self.assertEqual(result["intent_prediction"]["intent"], "mixed")
        self.assertEqual(len(result["tasks"]), 2)
        self.assertEqual(
            {task["sub_agent"] for task in result["tasks"]},
            {"customer_service", "manual"},
        )
        self.assertEqual(result["answer"], "订单运输中；断电后打开后盖更换滤网。")


if __name__ == "__main__":
    unittest.main()
