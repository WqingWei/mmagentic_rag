from __future__ import annotations

import asyncio
import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Coroutine, Dict

from .config import Settings
from .order_tool import MockOrderQueryTool


CUSTOMER_SERVICE_SYSTEM_PROMPT = """
你是电商平台的客服 Agent。回答要准确、简洁、友好，并与用户使用相同语言。

你可以处理订单查询、物流、退款、退换货、发票、售后和投诉问题。
当用户询问具体订单时：
1. 必须调用 query_order 工具，不得编造订单状态、商品、金额或物流信息。
2. 用户未提供订单号时，只询问订单号，不得猜测。
3. 工具返回未找到时，明确提示用户检查订单号。
4. 订单工具当前返回的是演示 Mock 数据；不要向用户泄露内部实现细节。

对于非订单问题，基于常见电商客服规范回答；不确定的平台政策要明确说明并建议转人工客服。
""".strip()


class AgentScopeCustomerServiceRuntime:
    """AgentScope ReAct runtime used by the customer-service sub-agent."""

    def __init__(
        self,
        settings: Settings,
        order_tool: MockOrderQueryTool | None = None,
    ):
        self.settings = settings
        self.order_tool = order_tool or MockOrderQueryTool()
        self._imports: Dict[str, Any] | None = None
        self._toolkit: Any = None

    def _load_agentscope(self) -> Dict[str, Any]:
        if self._imports is not None:
            return self._imports
        try:
            from agentscope.agent import ReActAgent
            from agentscope.formatter import OpenAIChatFormatter
            from agentscope.memory import InMemoryMemory
            from agentscope.message import Msg, TextBlock, ToolUseBlock
            from agentscope.model import OpenAIChatModel
            from agentscope.tool import Toolkit, ToolResponse
        except ImportError as exc:
            raise RuntimeError(
                "AgentScope 未安装，请执行 pip install -r requirements.txt。"
            ) from exc
        self._imports = {
            "ReActAgent": ReActAgent,
            "OpenAIChatFormatter": OpenAIChatFormatter,
            "InMemoryMemory": InMemoryMemory,
            "Msg": Msg,
            "TextBlock": TextBlock,
            "ToolUseBlock": ToolUseBlock,
            "OpenAIChatModel": OpenAIChatModel,
            "Toolkit": Toolkit,
            "ToolResponse": ToolResponse,
        }
        return self._imports

    @property
    def toolkit(self) -> Any:
        if self._toolkit is None:
            imports = self._load_agentscope()
            toolkit = imports["Toolkit"]()
            toolkit.register_tool_function(self.query_order)
            self._toolkit = toolkit
        return self._toolkit

    def query_order(self, order_id: str) -> Any:
        """Query an order by its exact order number.

        Args:
            order_id (`str`):
                The complete order number provided by the user.
        """
        imports = self._load_agentscope()
        result = self.order_tool.query(order_id)
        return imports["ToolResponse"](
            content=[
                imports["TextBlock"](
                    type="text",
                    text=json.dumps(result, ensure_ascii=False),
                ),
            ],
            metadata={"order_query": result},
        )

    async def lookup_order(self, order_id: str) -> Dict[str, Any]:
        """Execute the order lookup through AgentScope's registered Toolkit."""
        imports = self._load_agentscope()
        call = imports["ToolUseBlock"](
            type="tool_use",
            id=f"order-{order_id}",
            name="query_order",
            input={"order_id": order_id},
        )
        responses = await self.toolkit.call_tool_function(call)
        result: Dict[str, Any] | None = None
        async for response in responses:
            metadata = getattr(response, "metadata", None) or {}
            if isinstance(metadata.get("order_query"), dict):
                result = metadata["order_query"]
        return result or self.order_tool.query(order_id)

    async def answer(self, query: str) -> str:
        imports = self._load_agentscope()
        model = imports["OpenAIChatModel"](
            model_name=self.settings.llm_model,
            api_key=self.settings.dashscope_api_key,
            stream=False,
            client_kwargs={"base_url": self.settings.dashscope_base_url},
            generate_kwargs={"temperature": 0.1},
        )
        agent = imports["ReActAgent"](
            name="customer_service_agent",
            sys_prompt=CUSTOMER_SERVICE_SYSTEM_PROMPT,
            model=model,
            formatter=imports["OpenAIChatFormatter"](),
            toolkit=self.toolkit,
            memory=imports["InMemoryMemory"](),
            max_iters=self.settings.agentscope_max_iters,
        )
        disable_console = getattr(agent, "set_console_output_enabled", None)
        if callable(disable_console):
            disable_console(False)
        response = await agent(
            imports["Msg"](
                name="user",
                content=query,
                role="user",
            ),
        )
        answer = response.get_text_content().strip()
        if not answer:
            raise RuntimeError("AgentScope 客服 Agent 返回了空答案。")
        return answer

    def answer_sync(self, query: str) -> str:
        return self._run_sync(self.answer(query))

    def lookup_order_sync(self, order_id: str) -> Dict[str, Any]:
        return self._run_sync(self.lookup_order(order_id))

    @staticmethod
    def _run_sync(coroutine: Coroutine[Any, Any, Any]) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)

        # The public project API is synchronous. When it is invoked from an
        # existing event loop (for example, a notebook), run AgentScope on a
        # short-lived worker thread instead of nesting event loops.
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, coroutine).result()

    @staticmethod
    def query_language(query: str) -> str:
        return "zh" if re.search(r"[\u4e00-\u9fff]", query) else "en"
