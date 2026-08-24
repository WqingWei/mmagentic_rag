from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .agentscope_runtime import AgentScopeCustomerServiceRuntime
from .config import Settings
from .customer_qa_retriever import CustomerQARetriever
from .image_tool import ImageUnderstandingTool
from .order_tool import MockOrderQueryTool
from .schema import RetrievedChunk
from .tools import SearchManualTool


MAIN_SYSTEM_PROMPT = """
你是多模态电商客服系统的主 Agent，负责理解上下文、拆解问题、路由并整合答案。

可用工具：
- delegate_customer_service：订单、物流、退款、退换货、发票、投诉及售后政策。
- delegate_manual_qa：产品使用、安装、设置、清洁、维护、故障排查和安全说明。
- inspect_uploaded_images：仅当当前请求带图片时使用，用于读取图片中的产品、文字、故障码或损坏线索。

规则：
1. 每个用户诉求必须委派给合适的子 Agent，不能绕过工具自行回答。
2. 混合问题要拆成独立问题，并分别调用两个委派工具。
3. 有图片时，先调用 inspect_uploaded_images，再把图片事实和用户目标一起委派给子 Agent。
4. 订单号、物流状态、FAQ 政策和手册步骤只能采用子 Agent/工具返回的数据，禁止编造。
5. 最终合并各子 Agent 答案，保持用户语言；不要提及内部 Agent、工具、Mock、RAG 或框架名称。
""".strip()

CUSTOMER_SYSTEM_PROMPT = """
你是电商客服子 Agent，负责订单、物流、退款、退换货、发票、投诉和售后政策。

可用工具：query_order、search_customer_faq。
规则：
1. 具体订单查询必须调用 query_order；没有订单号时只询问订单号。
2. 政策或流程问题先调用 search_customer_faq。
3. 工具未命中时，可以给出谨慎的一般性建议，但不得编造平台政策或订单数据。
4. 使用和用户相同的语言，不要提及内部工具、Mock、检索或框架。
""".strip()

MANUAL_SYSTEM_PROMPT = """
你是产品手册问答子 Agent，只根据 search_manual 返回的说明书证据回答。

规则：
1. 回答前必须调用 search_manual；证据不足时可改写关键词后再次检索。
2. 不得使用未出现在检索结果中的型号、参数、步骤或安全结论。
3. 证据不足时明确说明，并询问产品名称/型号或建议查阅对应说明书。
4. 回答清晰、可执行，并保持用户语言；不要提及内部工具、RAG 或框架。
""".strip()


@dataclass
class AgentScopeExecutionContext:
    trace: List[str] = field(default_factory=list)
    tasks: List[Dict[str, Any]] = field(default_factory=list)
    task_results: List[Dict[str, Any]] = field(default_factory=list)
    image_ids: List[str] = field(default_factory=list)
    image_observation: Dict[str, Any] = field(default_factory=dict)

    def log(self, actor: str, action: str, detail: str = "") -> None:
        line = f"[{actor}] {action}"
        if detail:
            compact = " ".join(str(detail).split())
            line += f": {compact[:240]}" + ("..." if len(compact) > 240 else "")
        self.trace.append(line)

    def add_image_ids(self, image_ids: List[str]) -> None:
        for image_id in image_ids:
            if image_id and image_id not in self.image_ids:
                self.image_ids.append(image_id)


class AgentScopeMultiAgentApplication:
    """Complete AgentScope implementation used by CLI, API and chat entrypoints."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.order_tool = MockOrderQueryTool()
        self.framework = AgentScopeCustomerServiceRuntime(settings, self.order_tool)
        self.image_tool = ImageUnderstandingTool(settings)
        self._manual_search: SearchManualTool | None = None
        self._qa_retriever: CustomerQARetriever | None = None

    def close(self) -> None:
        if self._qa_retriever is not None:
            self._qa_retriever.close()

    @property
    def manual_search(self) -> SearchManualTool:
        if self._manual_search is None:
            self._manual_search = SearchManualTool(self.settings)
        return self._manual_search

    @property
    def qa_retriever(self) -> CustomerQARetriever:
        if self._qa_retriever is None:
            self._qa_retriever = CustomerQARetriever(self.settings)
        return self._qa_retriever

    def answer_sync(
        self,
        question: str,
        return_debug: bool = False,
        question_id: str = "",
        flow_log: bool = True,
        image_paths: List[str] | None = None,
        image_base64: List[str] | None = None,
        rewrite_history: List[Dict[str, Any]] | None = None,
        turn_mode: str = "auto_newline",
    ) -> Dict[str, Any]:
        return self.framework._run_sync(
            self.answer(
                question=question,
                return_debug=return_debug,
                question_id=question_id,
                flow_log=flow_log,
                image_paths=image_paths,
                image_base64=image_base64,
                rewrite_history=rewrite_history,
                turn_mode=turn_mode,
            )
        )

    async def answer(
        self,
        question: str,
        return_debug: bool = False,
        question_id: str = "",
        flow_log: bool = True,
        image_paths: List[str] | None = None,
        image_base64: List[str] | None = None,
        rewrite_history: List[Dict[str, Any]] | None = None,
        turn_mode: str = "auto_newline",
    ) -> Dict[str, Any]:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        if turn_mode not in {"auto_newline", "auto", "newline", "single", "single_turn"}:
            raise ValueError("turn_mode must be 'auto_newline' or 'single'")

        context = AgentScopeExecutionContext()
        context.log("AgentScope:main", "接收", f"question_id={question_id or '-'} query={question}")
        toolkit = self._build_main_toolkit(
            context=context,
            image_paths=image_paths or [],
            image_base64=image_base64 or [],
        )
        agent = self._build_react_agent(
            name="main_customer_agent",
            system_prompt=MAIN_SYSTEM_PROMPT,
            toolkit=toolkit,
        )
        user_content = self._build_user_content(
            question,
            history=rewrite_history or [],
            has_images=bool(image_paths or image_base64),
            turn_mode=turn_mode,
        )
        imports = self.framework._load_agentscope()
        response = await agent(imports["Msg"]("user", user_content, "user"))
        final_answer = response.get_text_content().strip()
        if not final_answer:
            raise RuntimeError("AgentScope 主 Agent 返回了空答案。")
        if not context.tasks:
            raise RuntimeError("AgentScope 主 Agent 未将请求委派给任何子 Agent。")
        if (image_paths or image_base64) and not context.image_observation:
            raise RuntimeError("AgentScope 主 Agent 未调用图片理解工具。")
        context.log("AgentScope:main", "输出", final_answer)
        if flow_log:
            for line in context.trace:
                print(line)
        return self._build_result(
            question=question,
            answer=final_answer,
            context=context,
            return_debug=return_debug,
        )

    def _build_main_toolkit(
        self,
        context: AgentScopeExecutionContext,
        image_paths: List[str],
        image_base64: List[str],
    ) -> Any:
        imports = self.framework._load_agentscope()
        toolkit = imports["Toolkit"]()

        async def delegate_customer_service(query: str) -> Any:
            """Delegate an order, logistics, refund, invoice or after-sales question.

            Args:
                query (`str`): A standalone customer-service question.
            """
            return await self._run_customer_agent(query, context)

        async def delegate_manual_qa(query: str) -> Any:
            """Delegate a product manual, operation or troubleshooting question.

            Args:
                query (`str`): A standalone product/manual question.
            """
            return await self._run_manual_agent(query, context)

        def inspect_uploaded_images() -> Any:
            """Inspect the images attached to the current user request."""
            observation = self.image_tool.observe(
                image_paths=image_paths,
                image_base64=image_base64,
            )
            context.image_observation = observation
            context.log(
                "AgentScope:image_tool",
                "观察",
                f"image_count={observation.get('image_count', 0)}",
            )
            return self._tool_response(observation, metadata={"image_observation": observation})

        toolkit.register_tool_function(delegate_customer_service)
        toolkit.register_tool_function(delegate_manual_qa)
        if image_paths or image_base64:
            toolkit.register_tool_function(inspect_uploaded_images)
        return toolkit

    async def _run_customer_agent(
        self,
        query: str,
        context: AgentScopeExecutionContext,
    ) -> Any:
        imports = self.framework._load_agentscope()
        toolkit = imports["Toolkit"]()
        tool_calls: List[Dict[str, Any]] = []

        def query_order(order_id: str) -> Any:
            """Query a specific order using its complete order number.

            Args:
                order_id (`str`): The complete order number from the user.
            """
            result = self.order_tool.query(order_id)
            tool_calls.append({"tool": "query_order", "result": result})
            return self._tool_response(result, metadata={"order_query": result})

        def search_customer_faq(faq_query: str) -> Any:
            """Search the customer-service FAQ knowledge base.

            Args:
                faq_query (`str`): The customer policy or process question to search.
            """
            if not self.settings.enable_customer_qa_retrieval:
                result = {"hit": False, "reason": "faq_retrieval_disabled"}
            else:
                result = self.qa_retriever.answer(faq_query)
            tool_calls.append({"tool": "search_customer_faq", "result": result})
            return self._tool_response(result)

        toolkit.register_tool_function(query_order)
        toolkit.register_tool_function(search_customer_faq)
        agent = self._build_react_agent(
            name="customer_service_agent",
            system_prompt=CUSTOMER_SYSTEM_PROMPT,
            toolkit=toolkit,
        )
        context.log("AgentScope:main", "委派", f"customer_service query={query}")
        response = await agent(imports["Msg"]("user", query, "user"))
        answer = response.get_text_content().strip()
        order_id = self.order_tool.extract_order_id(query)
        if order_id and not any(item.get("tool") == "query_order" for item in tool_calls):
            raise RuntimeError("客服子 Agent 未调用订单查询工具。")
        task_id = len(context.tasks) + 1
        route = self._customer_route(tool_calls)
        task = {
            "task_id": task_id,
            "turn_index": 1,
            "query": query,
            "sub_agent": "customer_service",
            "route_plan": {"intent": "customer_service", "source": "agentscope"},
        }
        context.tasks.append(task)
        context.task_results.append(
            {
                **task,
                "question_type": "customer_service",
                "route": route,
                "answer": answer,
                "answer_source": route,
                "tool_calls": tool_calls,
                "order_query": self._last_order_result(tool_calls),
            }
        )
        context.log("AgentScope:customer_service", "输出", f"route={route} answer={answer}")
        return self._tool_response(
            {"answer": answer, "route": route},
            metadata={"task_id": task_id, "route": route},
        )

    async def _run_manual_agent(
        self,
        query: str,
        context: AgentScopeExecutionContext,
    ) -> Any:
        imports = self.framework._load_agentscope()
        toolkit = imports["Toolkit"]()
        searches: List[Dict[str, Any]] = []

        def search_manual(search_query: str, top_k: int = 6) -> Any:
            """Search product manuals for grounded instructions and evidence.

            Args:
                search_query (`str`): Product, operation, symptom or error-code query.
                top_k (`int`): Maximum number of manual passages to return.
            """
            limit = min(max(int(top_k), 1), self.settings.rerank_top_k)
            chunks = self.manual_search.search(search_query, top_k=limit)
            payload = self._serialize_manual_chunks(chunks)
            searches.append({"query": search_query, "chunks": payload})
            context.add_image_ids(
                [image_id for chunk in chunks for image_id in chunk.image_ids]
            )
            return self._tool_response({"query": search_query, "chunks": payload})

        toolkit.register_tool_function(search_manual)
        agent = self._build_react_agent(
            name="manual_qa_agent",
            system_prompt=MANUAL_SYSTEM_PROMPT,
            toolkit=toolkit,
        )
        context.log("AgentScope:main", "委派", f"manual query={query}")
        response = await agent(imports["Msg"]("user", query, "user"))
        answer = response.get_text_content().strip()
        if not searches:
            raise RuntimeError("手册子 Agent 未调用说明书检索工具。")
        task_id = len(context.tasks) + 1
        task = {
            "task_id": task_id,
            "turn_index": 1,
            "query": query,
            "sub_agent": "manual",
            "route_plan": {"intent": "manual", "source": "agentscope"},
        }
        context.tasks.append(task)
        context.task_results.append(
            {
                **task,
                "question_type": "manual",
                "route": "agentscope_manual_rag",
                "answer": answer,
                "answer_source": "agentscope_manual_rag",
                "searches": searches,
            }
        )
        context.log("AgentScope:manual", "输出", f"searches={len(searches)} answer={answer}")
        return self._tool_response(
            {"answer": answer, "route": "agentscope_manual_rag"},
            metadata={"task_id": task_id, "route": "agentscope_manual_rag"},
        )

    def _build_react_agent(self, name: str, system_prompt: str, toolkit: Any) -> Any:
        imports = self.framework._load_agentscope()
        model = imports["OpenAIChatModel"](
            model_name=self.settings.llm_model,
            api_key=self.settings.dashscope_api_key,
            stream=False,
            client_kwargs={"base_url": self.settings.dashscope_base_url},
            generate_kwargs={"temperature": 0.0},
        )
        agent = imports["ReActAgent"](
            name=name,
            sys_prompt=system_prompt,
            model=model,
            formatter=imports["OpenAIChatFormatter"](),
            toolkit=toolkit,
            memory=imports["InMemoryMemory"](),
            max_iters=self.settings.agentscope_max_iters,
        )
        disable_console = getattr(agent, "set_console_output_enabled", None)
        if callable(disable_console):
            disable_console(False)
        return agent

    def _tool_response(
        self,
        value: Any,
        metadata: Dict[str, Any] | None = None,
    ) -> Any:
        imports = self.framework._load_agentscope()
        return imports["ToolResponse"](
            content=[
                imports["TextBlock"](
                    type="text",
                    text=json.dumps(value, ensure_ascii=False),
                )
            ],
            metadata=metadata,
        )

    @staticmethod
    def _build_user_content(
        question: str,
        history: List[Dict[str, Any]],
        has_images: bool,
        turn_mode: str,
    ) -> str:
        history_lines = [
            f"{item.get('role', 'unknown')}: {item.get('content', '')}"
            for item in history
            if str(item.get("content", "")).strip()
        ]
        parts = []
        if history_lines:
            parts.append("最近对话：\n" + "\n".join(history_lines))
        parts.append(f"当前用户请求：\n{question}")
        parts.append(f"当前请求包含图片：{'是' if has_images else '否'}")
        parts.append(f"换行处理模式：{turn_mode}")
        return "\n\n".join(parts)

    def _build_result(
        self,
        question: str,
        answer: str,
        context: AgentScopeExecutionContext,
        return_debug: bool,
    ) -> Dict[str, Any]:
        agents = [task["sub_agent"] for task in context.tasks]
        unique_agents = list(dict.fromkeys(agents))
        selected = unique_agents[0] if len(unique_agents) == 1 else "mixed" if unique_agents else "main"
        routes = [item.get("route", "") for item in context.task_results]
        route = routes[0] if len(routes) == 1 else "agentscope_multi_agent"
        order_queries = [
            item["order_query"]
            for item in context.task_results
            if item.get("order_query")
        ]
        result: Dict[str, Any] = {
            "framework": "agentscope",
            "question": question,
            "final_query": question,
            "rewritten_turns": [task["query"] for task in context.tasks] or [question],
            "main_plan": {
                "task_count": len(context.tasks),
                "selected_sub_agent": selected,
                "orchestrator": "AgentScope ReActAgent",
            },
            "tasks": context.tasks,
            "task_results": context.task_results if return_debug else [
                self._compact_task_result(item) for item in context.task_results
            ],
            "turn_results": [
                {
                    "turn_index": 1,
                    "original_query": question,
                    "task_ids": [task["task_id"] for task in context.tasks],
                    "question_types": [item["question_type"] for item in context.task_results],
                    "answer": answer,
                    "image_ids": context.image_ids,
                    "answer_sources": [item["answer_source"] for item in context.task_results],
                }
            ],
            "answers": [answer],
            "answer": answer,
            "image_ids": context.image_ids,
            "question_type": selected,
            "route": route,
            "route_plan": {
                "intent": selected,
                "source": "agentscope",
                "task_routes": routes,
            },
            "selected_sub_agent": selected,
            "sub_agent_result_type": selected,
            "answer_sources": [item["answer_source"] for item in context.task_results],
            "order_queries": order_queries,
            "agent_trace": context.trace,
        }
        if len(order_queries) == 1:
            result["order_query"] = order_queries[0]
        if context.image_observation:
            result["image_observation"] = context.image_observation
        return result

    @staticmethod
    def _compact_task_result(result: Dict[str, Any]) -> Dict[str, Any]:
        keys = (
            "task_id",
            "turn_index",
            "query",
            "sub_agent",
            "question_type",
            "route",
            "answer",
            "answer_source",
            "order_query",
        )
        return {key: result[key] for key in keys if result.get(key) not in (None, {}, [])}

    @staticmethod
    def _serialize_manual_chunks(chunks: List[RetrievedChunk]) -> List[Dict[str, Any]]:
        return [
            {
                "chunk_id": chunk.chunk_id,
                "manual_id": chunk.manual_id,
                "section_path": chunk.section_path,
                "text": chunk.text,
                "image_ids": chunk.image_ids,
                "score": chunk.rerank_score if chunk.rerank_score is not None else chunk.score,
            }
            for chunk in chunks
        ]

    @staticmethod
    def _customer_route(tool_calls: List[Dict[str, Any]]) -> str:
        names = [item.get("tool") for item in tool_calls]
        if "query_order" in names:
            return "mock_order_query"
        if "search_customer_faq" in names:
            faq_calls = [item for item in tool_calls if item.get("tool") == "search_customer_faq"]
            if any(item.get("result", {}).get("hit") for item in faq_calls):
                return "customer_qa_retrieval"
        return "agentscope_customer_service"

    @staticmethod
    def _last_order_result(tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        for item in reversed(tool_calls):
            if item.get("tool") == "query_order":
                return item.get("result", {})
        return {}
