from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .agentscope_runtime import AgentScopeCustomerServiceRuntime
from .config import Settings
from .customer_qa_retriever import CustomerQARetriever
from .dag_planner import DAGNode, ExecutionPlan, build_simple_plan, plan_from_json
from .image_tool import ImageUnderstandingTool
from .intent_model import (
    IntentClassifier,
    IntentPrediction,
    IntentType,
    build_intent_classifier,
)
from .order_tool import MockOrderQueryTool
from .schema import RetrievedChunk
from .tools import SearchManualTool


PLANNER_SYSTEM_PROMPT = """
你是多模态客服系统的 Planner Agent。把复杂、多轮、混合或图片请求规划成可执行 DAG。

节点类型：
- image_understanding：理解当前请求中的图片。
- knowledge：产品知识、说明书、操作和故障排查。
- customer_service：订单、物流、退款、退换货、发票、投诉和售后。
- clarification：关键信息不足时追问。

如果 has_images=true，你必须先调用 inspect_request_images，根据图片观察判断应该路由到知识、客服还是澄清节点；
之后必须调用 submit_plan，参数 plan_json 是 JSON 字符串，结构如下：
{"intent":"mixed","reason":"...","summary_required":true,"nodes":[
  {"node_id":"image_1","node_type":"image_understanding","query":"...","depends_on":[]},
  {"node_id":"task_1","node_type":"knowledge","query":"...","depends_on":["image_1"]}
]}

规则：节点 ID 唯一；依赖必须存在且无环；图片节点必须先于依赖图片的任务；独立诉求拆成并行节点；不要回答问题。
""".strip()

SUMMARY_SYSTEM_PROMPT = """
你是 Summary Agent。根据用户原始请求、意图分类、DAG 和各节点结果生成最终回复。
必须忠实采用节点结果，不得编造订单、政策或说明书信息。合并重复内容，保留必要步骤；
若存在 clarification 结果，只提出最少且明确的补充问题。保持用户语言，不提及内部 Agent、DAG、工具、Mock 或框架。
""".strip()

CLARIFICATION_SYSTEM_PROMPT = """
你是澄清 Agent。识别继续处理请求所缺少的最少信息，只提出 1 至 2 个具体问题。
不要猜测产品、订单、故障或用户目标，不要直接给解决方案，保持用户语言。
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
    intent_prediction: Dict[str, Any] = field(default_factory=dict)
    execution_plan: Dict[str, Any] = field(default_factory=dict)
    node_outputs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

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

    def __init__(
        self,
        settings: Settings,
        intent_classifier: IntentClassifier | None = None,
    ):
        self.settings = settings
        self.order_tool = MockOrderQueryTool()
        self.framework = AgentScopeCustomerServiceRuntime(settings, self.order_tool)
        self.image_tool = ImageUnderstandingTool(settings)
        self.intent_classifier = intent_classifier or build_intent_classifier(
            settings,
            self.framework,
        )
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

        history = rewrite_history or []
        has_images = bool(image_paths or image_base64)
        context = AgentScopeExecutionContext()
        context.log("AgentScope:intent", "接收", f"question_id={question_id or '-'} query={question}")
        prediction = await self.intent_classifier.classify(
            question,
            history=history,
            has_images=has_images,
        )
        context.intent_prediction = prediction.to_dict()
        context.log(
            "AgentScope:intent",
            "输出",
            f"intent={prediction.intent.value} confidence={prediction.confidence:.3f} "
            f"planner={prediction.needs_planner} source={prediction.source}",
        )
        plan = await self._create_execution_plan(
            question=question,
            history=history,
            has_images=has_images,
            image_paths=image_paths or [],
            image_base64=image_base64 or [],
            prediction=prediction,
            context=context,
        )
        context.execution_plan = plan.to_dict()
        await self._execute_plan(
            plan=plan,
            question=question,
            context=context,
            image_paths=image_paths or [],
            image_base64=image_base64 or [],
        )
        final_answer = await self._run_summary_agent(question, context)
        context.log("AgentScope:summary", "输出", final_answer)
        if flow_log:
            for line in context.trace:
                print(line)
        return self._build_result(
            question=question,
            answer=final_answer,
            context=context,
            return_debug=return_debug,
        )

    async def _create_execution_plan(
        self,
        question: str,
        history: List[Dict[str, Any]],
        has_images: bool,
        image_paths: List[str],
        image_base64: List[str],
        prediction: IntentPrediction,
        context: AgentScopeExecutionContext,
    ) -> ExecutionPlan:
        if not prediction.needs_planner:
            plan = build_simple_plan(prediction, question, has_images=has_images)
            context.log("AgentScope:planner", "快速规划", plan.to_dict())
            return plan

        imports = self.framework._load_agentscope()
        toolkit = imports["Toolkit"]()
        submitted: Dict[str, str] = {}
        inspected = {"done": False}

        def inspect_request_images() -> Any:
            """Inspect all images attached to the current request before planning."""
            observation = self.image_tool.observe(
                image_paths=image_paths,
                image_base64=image_base64,
                user_question=question,
            )
            context.image_observation = observation
            inspected["done"] = True
            context.log("AgentScope:planner", "图片理解", observation)
            return self._tool_response(observation)

        def submit_plan(plan_json: str) -> Any:
            """Submit the final executable DAG plan as JSON.

            Args:
                plan_json (`str`): JSON object containing intent, nodes and dependencies.
            """
            plan = plan_from_json(plan_json, prediction.intent)
            submitted["plan"] = plan_json
            return self._tool_response({"accepted": True, "plan": plan.to_dict()})

        toolkit.register_tool_function(inspect_request_images)
        toolkit.register_tool_function(submit_plan)
        agent = self._build_react_agent(
            name="planner_agent",
            system_prompt=PLANNER_SYSTEM_PROMPT,
            toolkit=toolkit,
        )
        planner_input = {
            "query": question,
            "history": history,
            "has_images": has_images,
            "intent_prediction": prediction.to_dict(),
        }
        context.log("AgentScope:planner", "规划", planner_input)
        await agent(
            imports["Msg"](
                "user",
                json.dumps(planner_input, ensure_ascii=False),
                "user",
            )
        )
        if not submitted.get("plan"):
            raise RuntimeError("Planner Agent 未提交 DAG 计划。")
        if has_images and not inspected["done"]:
            raise RuntimeError("Planner Agent 未在规划前调用图片理解工具。")
        plan = plan_from_json(submitted["plan"], prediction.intent)
        if has_images and not any(
            node.node_type == "image_understanding" for node in plan.nodes
        ):
            raise RuntimeError("带图片的请求缺少 image_understanding DAG 节点。")
        context.log("AgentScope:planner", "输出", plan.to_dict())
        return plan

    async def _execute_plan(
        self,
        plan: ExecutionPlan,
        question: str,
        context: AgentScopeExecutionContext,
        image_paths: List[str],
        image_base64: List[str],
    ) -> None:
        for node in plan.topological_nodes():
            query = self._node_query(node, context)
            context.log(
                "AgentScope:dag",
                "执行",
                f"node={node.node_id} type={node.node_type} depends_on={node.depends_on}",
            )
            if node.node_type == "image_understanding":
                output = context.image_observation or self.image_tool.observe(
                    image_paths=image_paths,
                    image_base64=image_base64,
                    user_question=question,
                )
                context.image_observation = output
                context.node_outputs[node.node_id] = {
                    "node_type": node.node_type,
                    "result": output,
                }
            elif node.node_type == "customer_service":
                await self._run_customer_agent(query, context, node=node)
            elif node.node_type == "knowledge":
                await self._run_manual_agent(query, context, node=node)
            elif node.node_type == "clarification":
                await self._run_clarification_agent(query, context, node=node)

    async def _run_summary_agent(
        self,
        question: str,
        context: AgentScopeExecutionContext,
    ) -> str:
        imports = self.framework._load_agentscope()
        agent = self._build_react_agent(
            name="summary_agent",
            system_prompt=SUMMARY_SYSTEM_PROMPT,
            toolkit=imports["Toolkit"](),
        )
        payload = {
            "question": question,
            "intent": context.intent_prediction,
            "plan": context.execution_plan,
            "task_results": context.task_results,
            "image_observation": context.image_observation,
        }
        context.log("AgentScope:summary", "汇总", f"tasks={len(context.task_results)}")
        response = await agent(
            imports["Msg"]("user", json.dumps(payload, ensure_ascii=False), "user")
        )
        answer = response.get_text_content().strip()
        if not answer:
            raise RuntimeError("Summary Agent 返回了空答案。")
        return answer

    async def _run_clarification_agent(
        self,
        query: str,
        context: AgentScopeExecutionContext,
        node: DAGNode,
    ) -> None:
        imports = self.framework._load_agentscope()
        agent = self._build_react_agent(
            name="clarification_agent",
            system_prompt=CLARIFICATION_SYSTEM_PROMPT,
            toolkit=imports["Toolkit"](),
        )
        response = await agent(imports["Msg"]("user", query, "user"))
        answer = response.get_text_content().strip()
        task = self._task_from_node(node, "clarification")
        context.tasks.append(task)
        result = {
            **task,
            "question_type": "clarification",
            "route": "agentscope_clarification",
            "answer": answer,
            "answer_source": "agentscope_clarification",
        }
        context.task_results.append(result)
        context.node_outputs[node.node_id] = result
        context.log("AgentScope:clarification", "输出", answer)

    @staticmethod
    def _task_from_node(
        node: DAGNode,
        sub_agent: str,
        query: str | None = None,
    ) -> Dict[str, Any]:
        return {
            "task_id": node.node_id,
            "turn_index": 1,
            "dag_node_id": node.node_id,
            "depends_on": node.depends_on,
            "query": query or node.query,
            "sub_agent": sub_agent,
            "route_plan": {
                "intent": sub_agent,
                "source": "agentscope_dag_planner",
            },
        }

    @staticmethod
    def _node_query(node: DAGNode, context: AgentScopeExecutionContext) -> str:
        dependency_context = []
        for dependency in node.depends_on:
            output = context.node_outputs.get(dependency, {})
            if output.get("node_type") == "image_understanding":
                observation = output.get("result", {})
                dependency_context.append(
                    {
                        "dependency": dependency,
                        "image_observation": {
                            key: observation.get(key)
                            for key in (
                                "product_type",
                                "brand_or_model",
                                "visible_parts",
                                "visible_error_code",
                                "ocr_text",
                                "vlm_summary",
                                "likely_intent",
                            )
                            if observation.get(key) not in (None, "", [])
                        },
                    }
                )
            else:
                dependency_context.append(
                    {
                        "dependency": dependency,
                        "answer": output.get("answer", ""),
                        "route": output.get("route", ""),
                    }
                )
        if not dependency_context:
            return node.query
        return (
            f"{node.query}\n\n上游节点上下文：\n"
            + json.dumps(dependency_context, ensure_ascii=False)
        )

    async def _run_customer_agent(
        self,
        query: str,
        context: AgentScopeExecutionContext,
        node: DAGNode,
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
        context.log("AgentScope:dag", "路由", f"customer_service query={query}")
        response = await agent(imports["Msg"]("user", query, "user"))
        answer = response.get_text_content().strip()
        order_id = self.order_tool.extract_order_id(query)
        if order_id and not any(item.get("tool") == "query_order" for item in tool_calls):
            raise RuntimeError("客服子 Agent 未调用订单查询工具。")
        route = self._customer_route(tool_calls)
        task = self._task_from_node(node, "customer_service", query=query)
        task_id = task["task_id"]
        context.tasks.append(task)
        result = {
            **task,
            "question_type": "customer_service",
            "route": route,
            "answer": answer,
            "answer_source": route,
            "tool_calls": tool_calls,
            "order_query": self._last_order_result(tool_calls),
        }
        context.task_results.append(result)
        context.node_outputs[node.node_id] = result
        context.log("AgentScope:customer_service", "输出", f"route={route} answer={answer}")
        return self._tool_response(
            {"answer": answer, "route": route},
            metadata={"task_id": task_id, "route": route},
        )

    async def _run_manual_agent(
        self,
        query: str,
        context: AgentScopeExecutionContext,
        node: DAGNode,
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
        context.log("AgentScope:dag", "路由", f"manual query={query}")
        response = await agent(imports["Msg"]("user", query, "user"))
        answer = response.get_text_content().strip()
        if not searches:
            raise RuntimeError("手册子 Agent 未调用说明书检索工具。")
        task = self._task_from_node(node, "manual", query=query)
        task_id = task["task_id"]
        context.tasks.append(task)
        result = {
            **task,
            "question_type": "manual",
            "route": "agentscope_manual_rag",
            "answer": answer,
            "answer_source": "agentscope_manual_rag",
            "searches": searches,
        }
        context.task_results.append(result)
        context.node_outputs[node.node_id] = result
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
            "intent_prediction": context.intent_prediction,
            "execution_plan": context.execution_plan,
            "rewritten_turns": [task["query"] for task in context.tasks] or [question],
            "main_plan": {
                "task_count": len(context.tasks),
                "selected_sub_agent": selected,
                "orchestrator": "AgentScope DAG Planner Agent",
                "dag": context.execution_plan,
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
            "question_type": context.intent_prediction.get("intent", selected),
            "route": route,
            "route_plan": {
                "intent": context.intent_prediction.get("intent", selected),
                "source": context.intent_prediction.get("source", "agentscope"),
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
