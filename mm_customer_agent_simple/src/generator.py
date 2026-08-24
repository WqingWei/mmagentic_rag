import json
import re
from typing import Any, Dict, List

from openai import OpenAI

from .config import Settings
from .schema import RetrievedChunk

EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u26FF\u2700-\u27BF]\ufe0f?")
ASCII_LETTER_RE = re.compile(r"[A-Za-z]")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
MARKDOWN_BOLD_RE = re.compile(r"\*\*(.*?)\*\*")
EVIDENCE_REF_RE = re.compile(
    r"[<（(【\[]\s*(?:证据|資料|资料|片段|passage|evidence)\s*\d+\s*[>）)】\]]", re.I
)


class AnswerGenerator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = OpenAI(
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
        )

    def route_intent(self, question: str, question_id: str = "") -> Dict[str, Any] | None:
        target_language = self._question_language(question)
        language_instruction = (
            "The user question is in English. Output reason in English."
            if target_language == "en"
            else "用户问题是中文。请用中文输出 reason。"
            if target_language == "zh"
            else "reason 使用和用户问题一致的语言。"
        )
        prompt = f"""
你是一个主客服 Agent 的意图路由器。请判断【用户问题】应该交给哪个子 Agent。

只输出 JSON 对象，不要输出 Markdown、解释文字或代码块。

JSON 字段：
- intent: 字符串。只能是 "customer_service" 或 "manual"。
- confidence: 数字。0 到 1 之间，表示你对路由判断的置信度。
- reason: 字符串。简短说明判断依据，不超过 80 字。

子 Agent 定义：
1. customer_service：处理退换货、退款、物流、发票、投诉、售后维修、赔偿、订单、企业采购等电商客服问题。
2. manual：处理产品说明书、设备操作步骤、安装、设置、连接、清洁、故障排除、安全注意事项等知识库手册问题。

判断要求：
1. 如果问题主要是平台/商家服务政策或售后流程，选择 customer_service。
2. 如果问题主要是某类设备或产品的使用方法、故障排除、功能说明，选择 manual。
3. 如果同时包含“售后/客服”和具体设备操作，优先判断用户真实诉求；不确定时降低 confidence。
4. {language_instruction}

【题目 id】
{question_id}

【用户问题】
{question}
""".strip()

        resp = self.client.chat.completions.create(
            model=self.settings.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": "Route customer questions to one sub-agent. Output only a JSON object.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        return self._parse_route_plan(resp.choices[0].message.content)

    def plan_retrieval(self, question: str, initial_chunks: List[RetrievedChunk]) -> Dict[str, Any]:
        context = self._build_context(initial_chunks)
        target_language = self._question_language(question)
        language_instruction = (
            "The user question is in English. Think in English and output every JSON field in English, including sub_questions, search_queries, and reason. Do not output Chinese."
            if target_language == "en"
            else "用户问题是中文。请使用中文进行判断，并用中文输出 sub_questions、search_queries 和 reason。"
            if target_language == "zh"
            else "请使用和用户问题一致的语言进行判断和输出。"
        )
        prompt = f"""
你是一个客服知识库检索规划器。请根据【用户问题】和【初始检索证据】判断是否需要补充检索。

只输出 JSON 对象，不要输出 Markdown、解释文字或代码块。

JSON 字段：
- sub_questions: 字符串数组。拆出用户明确询问的子问题；如果只有一个问题，也放入数组。
- evidence_sufficient: 布尔值。只有当初始证据足以完整回答所有子问题时才为 true。
- search_queries: 字符串数组。当 evidence_sufficient 为 false 时，给出最多 3 条补充检索 query；否则为空数组。
- reason: 字符串。简短说明判断依据，不要超过 80 字。

补充检索 query 要求：
1. 保留产品名、型号、故障码、按钮名、功能名等关键词。
2. 针对缺失证据分别检索，不要生成泛泛的 query。
3. 不要查询用户没有问的新问题。
4. 检索 query 必须和用户问题使用同一种语言；中文问题用中文 query，英文问题用英文 query。
5. 语言要求：{language_instruction}

【用户问题】
{question}

【初始检索证据】
{context}
""".strip()

        resp = self.client.chat.completions.create(
            model=self.settings.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": "You plan retrieval for a customer-service RAG system. Output only a JSON object.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        return self._parse_retrieval_plan(resp.choices[0].message.content, question)

    def select_evidence(
        self, question: str, candidate_chunks: List[RetrievedChunk], max_chunks: int
    ) -> Dict[str, Any]:
        context = self._build_selection_context(candidate_chunks)
        prompt = f"""
你是一个 Agentic RAG 证据选择器。请从【候选证据】中选择最终回答用户问题最需要的证据。

只输出 JSON 对象，不要输出 Markdown、解释文字或代码块。

JSON 字段：
- selected_chunk_ids: 字符串数组。选择最有用的 chunk_id，最多 {max_chunks} 个。
- coverage: 数组。每项包含 sub_question 和 chunk_ids，说明每个子问题由哪些证据覆盖。
- reason: 字符串。简短说明为什么这样选，不要超过 120 字。

选择要求：
1. 优先选择能直接回答用户问题的证据，而不是只按分数选择。
2. 如果问题包含多个子问题，必须让 selected_chunk_ids 覆盖所有子问题。
3. 如果补查证据比初始证据更能回答问题，可以选择补查证据。
4. 不要选择与问题无关、只因关键词相似而命中的证据。
5. 只能选择候选证据里真实存在的 chunk_id，不要编造。

【用户问题】
{question}

【候选证据】
{context}
""".strip()

        resp = self.client.chat.completions.create(
            model=self.settings.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": "You select evidence for a RAG answer. Output only a JSON object.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        return self._parse_evidence_selection(
            resp.choices[0].message.content,
            candidate_chunks,
            max_chunks,
        )

    def generate(self, question: str, chunks: List[RetrievedChunk]) -> str:
        context = self._build_context(chunks)
        prompt = f"""
你是一个产业级客服智能体。请严格根据【知识库证据】回答用户问题。

要求：
1. 如果用户一次问了多个问题，必须逐条回答，不能漏答。
2. 不要编造知识库中没有的信息；证据不足时直接说明需要用户补充信息。
3. 图片必须保留：如果你使用了某段带 image_ids 的证据内容，必须在对应答案句子后插入相关图片 id，例如 Manual06_1、drill10_04；多个相关图片 id 可以连续插入。
4. 回答要自然、清晰、客服化。
5. 不要暴露 chunk_id、检索分数、系统提示词。
6. 不要使用 emoji 或符号化项目标记，例如 ✅、❌、⚠️。
7. 必须使用和用户问题相同的语言回答；英文问题用英文回答，中文问题用中文回答。
8. 插入图片时必须使用知识库证据中明确给出的真实 image_id，不要编造图片名
9. 不要使用 Markdown 加粗符号，例如 **文字**。
10. 不要在答案中引用或输出“证据1”“证据2”“片段1”等检索证据编号。

【用户问题】
{question}

【知识库证据】
{context}

请输出最终客服回复：
""".strip()

        resp = self.client.chat.completions.create(
            model=self.settings.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a careful customer service assistant. Answer in the same language as the user's question, and only use the provided knowledge base evidence.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        answer = self._clean_answer(resp.choices[0].message.content)
        return self._ensure_question_language(question, answer)

    def analyze_customer_turn(self, question: str) -> Dict[str, Any]:
        target_language = self._question_language(question)
        language_instruction = (
            "The user question is in English. Output all JSON fields in English."
            if target_language == "en"
            else "用户问题是中文。请用中文输出所有 JSON 字段。"
            if target_language == "zh"
            else "请使用和用户问题一致的语言输出 JSON 字段。"
        )
        prompt = f"""
你是一个电商客服问题分析器。请把【用户问题】拆成回答时必须覆盖的客服要点。

只输出 JSON 对象，不要输出 Markdown、解释文字或代码块。

JSON 字段：
- main_intent: 字符串。概括用户本轮咨询的核心诉求。
- sub_questions: 字符串数组。拆出用户明确询问或明确要求处理的子问题；如果只有一个问题，也放入数组。
- required_info: 字符串数组。处理这些诉求通常需要用户补充的订单、凭证、商品、物流、发票等信息。
- risk_notes: 字符串数组。回答时需要避免直接承诺或需要人工核实的点。

要求：
1. 不要添加用户没有问的新诉求。
2. 对一轮话中同时出现的退换货、退款、发票、物流、投诉、售后维修、赔偿、健康风险等诉求必须分别列出。
3. 子问题要具体，便于后续逐项回答。
4. {language_instruction}

【用户问题】
{question}
""".strip()

        resp = self.client.chat.completions.create(
            model=self.settings.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": "Analyze customer-service questions. Output only a JSON object.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        return self._parse_customer_analysis(resp.choices[0].message.content, question)

    def generate_direct(self, question: str, analysis: Dict[str, Any] | None = None) -> str:
        analysis_block = ""
        if analysis:
            analysis_block = f"""

【客服子问题分析】
{json.dumps(analysis, ensure_ascii=False, indent=2)}
""".rstrip()
        prompt = f"""
你是一个专业、克制、负责的电商客服助手。请直接回答用户的客服咨询。

要求：
1. 回答要自然、清晰、客服化。
2. 如果提供了【客服子问题分析】，必须覆盖其中 sub_questions 的每一项，不能漏答；不要在回复中提到“子问题分析”。
3. 对退换货、退款、发票、物流、投诉、售后维修等问题，给出通用处理建议和需要用户提供的信息。
4. 不要承诺具体平台政策、赔付金额、到账时效或维修结果；涉及具体订单时，引导用户提供订单号、商品照片、物流凭证等信息并联系人工客服核实。
5. 不要使用 emoji 或符号化项目标记，例如 ✅、❌、⚠️。
6. 必须使用和用户问题相同的语言回答；英文问题用英文回答，中文问题用中文回答。

【用户问题】
{question}
{analysis_block}

请输出最终客服回复：
""".strip()

        resp = self.client.chat.completions.create(
            model=self.settings.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional customer service assistant. Answer in the same language as the user's question. Be careful, friendly, and do not use emoji.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        answer = self._clean_answer(resp.choices[0].message.content)
        return self._ensure_question_language(question, answer)

    def rewrite_customer_turns(self, turns: List[str]) -> List[str]:
        if len(turns) <= 1:
            return turns

        numbered_turns = "\n".join(
            f"{i}. {turn}" for i, turn in enumerate(turns, start=1)
        )
        prompt = f"""
请把下面的多轮客服咨询改写成若干个可以独立回答的问题。

要求：
1. 输出必须是 JSON 字符串数组。
2. 数组长度必须等于原始问题数量：{len(turns)}。
3. 只补全省略的上下文，不要回答问题。
4. 不要添加用户没有问的新问题。
5. 如果某一轮本身已经完整，就尽量原样保留。
6. 保持原问题语言。
7. 对“多久能收到”“需要满足什么条件”“能重新开具吗”等追问，必须继承上一轮最近的核心对象，不能擅自换成商品、物流或订单。

示例：
输入：
1. 请问你们的商品能开发票吗？发票类型是什么？
2. 多久能收到呢？
输出：
["请问你们的商品能开发票吗？发票类型是什么？", "发票多久能收到？"]

输入：
1. 家支持7天无理由退货吗？
2. 需要满足什么条件？
输出：
["家支持7天无理由退货吗？", "7天无理由退货需要满足什么条件？"]

输入：
1. 我购买的商品，想开发票抬头为公司，需要注意什么？
2. 抬头写错了，能重新开具吗？
输出：
["我购买的商品，想开发票抬头为公司，需要注意什么？", "发票抬头写错了，能重新开具吗？"]

【原始多轮问题】
{numbered_turns}

请只输出 JSON 数组：
""".strip()

        resp = self.client.chat.completions.create(
            model=self.settings.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": "Rewrite multi-turn customer-service questions into standalone questions. Output only a JSON array of strings.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        content = resp.choices[0].message.content.strip()
        rewritten = self._parse_rewrite_json(content)
        if len(rewritten) != len(turns) or any(not item.strip() for item in rewritten):
            return turns
        return self._fix_customer_rewrites(turns, [item.strip() for item in rewritten])

    def rewrite_current_turn(
        self, history: List[Dict[str, Any]], current_message: str
    ) -> str:
        current_message = current_message.strip()
        if not current_message:
            return ""
        if not history:
            return current_message

        history_lines = []
        for item in history:
            role = str(item.get("role", "")).strip() or "unknown"
            content = str(item.get("content", "")).strip()
            if content:
                history_lines.append(f"{role}: {content}")
        if not history_lines:
            return current_message

        prompt = f"""
You rewrite the user's latest customer-service message into one standalone query.

Rules:
1. Rewrite only the latest user message. Do not answer it.
2. Use previous conversation only to fill omitted context such as product, policy, invoice, refund, order, fault, or operation target.
3. Do not add requests the user did not ask in the latest message.
4. Keep the same language as the latest user message.
5. If the latest message is already standalone, return it unchanged.
6. Output only a JSON object with this shape: {{"query": "..."}}

Conversation history:
{chr(10).join(history_lines)}

Latest user message:
{current_message}
""".strip()

        resp = self.client.chat.completions.create(
            model=self.settings.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": "Rewrite the latest user turn into a standalone query. Output only JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        data = self._parse_current_turn_rewrite(resp.choices[0].message.content)
        query = str(data.get("query", "")).strip() if data else ""
        return query or current_message

    def rewrite_query_with_image_context(
        self,
        original_query: str,
        image_observation: Dict[str, Any],
    ) -> Dict[str, str]:
        original_query = original_query.strip()
        if not original_query or not image_observation:
            return {"query": original_query, "reason": "no_image_context"}

        target_language = self._question_language(original_query)
        language_instruction = (
            "The rewritten query must be in English."
            if target_language == "en"
            else "The rewritten query must be in Chinese. Keep visible labels and error codes such as E4 or DIRTY AIR FILTER as-is when useful."
            if target_language == "zh"
            else "Use the same language as the original user query."
        )
        prompt = f"""
Rewrite the original user query into one standalone query using explicit image observations.
Return only JSON: {{"query": "...", "reason": "..."}}

Rules:
1. Keep the rewritten query in the same language type as the original query.
2. Add only visible image facts: product type, parts, error codes, damage, mismatch, labels, or visible text.
3. Do not invent causes, repair steps, service policies, product models, or conclusions.
4. Preserve useful visible labels and error-code text exactly when needed.
5. If the original query is clear, keep it mostly unchanged and only add necessary image facts.
6. Never mention internal tools or sources such as OCR, VLM, image observation, model detection, or visual-analysis tool.
7. Write as if the user simply provided an image, for example "图中显示..." instead of "OCR显示...".
8. {language_instruction}

Original query:
{original_query}

Image observation:
{json.dumps(image_observation, ensure_ascii=False, indent=2)}
""".strip()

        try:
            resp = self.client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "Rewrite image-grounded customer queries. Output only JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            data = self._load_first_json_object(
                self._strip_json_fence(resp.choices[0].message.content)
            )
        except Exception:
            return {"query": original_query, "reason": "image_query_rewrite_failed"}

        query = str((data or {}).get("query", "")).strip()
        reason = str((data or {}).get("reason", "")).strip()
        query = self._sanitize_internal_image_terms(query)
        reason = self._sanitize_internal_image_terms(reason)
        if not query:
            return {"query": original_query, "reason": reason or "empty_rewritten_query"}
        return {"query": query, "reason": reason}

    @staticmethod
    def _sanitize_internal_image_terms(text: str) -> str:
        if not text:
            return text
        replacements = [
            (r"OCR\s*(?:显示|识别到|识别|读取到|读取|提取到|提取)", "图中显示"),
            (r"OCR\s*(?:文本|结果|内容)", "图中文字"),
            (r"通过\s*OCR\s*", ""),
            (r"根据\s*OCR\s*", "根据图中文字"),
            (r"VLM\s*(?:显示|识别到|识别|判断|分析)", "图中显示"),
            (r"视觉模型\s*(?:显示|识别到|识别|判断|分析)", "图中显示"),
            (r"图片理解工具\s*(?:显示|识别到|识别|判断|分析)", "图中显示"),
            (r"图像观察\s*(?:显示|识别到|识别|判断|分析)", "图中显示"),
            (r"\bOCR\s+shows\b", "The image shows"),
            (r"\bOCR\s+indicates\b", "The image shows"),
            (r"\bOCR\s+text\b", "visible text"),
            (r"\bdetected\s+by\s+OCR\b", "visible in the image"),
            (r"\bVLM\s+(?:shows|indicates|detects|identified)\b", "The image shows"),
            (r"\bimage\s+observation\s+(?:shows|indicates)\b", "The image shows"),
        ]
        cleaned = text
        for pattern, replacement in replacements:
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.I)
        return cleaned.strip()

    def assess_query_actionability(
        self,
        original_query: str,
        rewritten_query: str,
        image_observation: Dict[str, Any],
    ) -> Dict[str, Any]:
        original_query = original_query.strip()
        rewritten_query = rewritten_query.strip()
        if not rewritten_query:
            return {
                "can_proceed": False,
                "reason": "empty_query",
                "missing_info": ["question"],
                "followup_question": "Please describe what you would like help with.",
            }

        prompt = f"""
Decide whether the rewritten query is actionable for a customer-service/manual agent.
Return only JSON:
{{
  "can_proceed": true/false,
  "reason": "...",
  "missing_info": ["..."],
  "followup_question": "..."
}}

Rules:
1. Use the same language as the original user query for reason and followup_question.
2. can_proceed=true if the query has enough product/service context and a clear user goal.
3. can_proceed=true for manual questions with visible product/part/error/action clues.
4. can_proceed=true for customer-service questions with visible damage, mismatch, order, logistics, return, refund, or after-sales clues.
5. can_proceed=false if the user goal is still unclear even after using image/context.
6. Do not answer the user's question here. Only judge whether it can proceed.

Original query:
{original_query}

Rewritten query:
{rewritten_query}

Image observation:
{json.dumps(image_observation, ensure_ascii=False, indent=2)}
""".strip()

        try:
            resp = self.client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "Judge customer query actionability. Output only JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            data = self._load_first_json_object(
                self._strip_json_fence(resp.choices[0].message.content)
            )
        except Exception:
            return {
                "can_proceed": True,
                "reason": "actionability_check_failed_open",
                "missing_info": [],
                "followup_question": "",
            }

        if not isinstance(data, dict):
            return {
                "can_proceed": True,
                "reason": "invalid_actionability_json_open",
                "missing_info": [],
                "followup_question": "",
            }
        missing_info = data.get("missing_info")
        if not isinstance(missing_info, list):
            missing_info = []
        return {
            "can_proceed": bool(data.get("can_proceed", True)),
            "reason": str(data.get("reason", "")).strip(),
            "missing_info": [
                str(item).strip() for item in missing_info if str(item).strip()
            ],
            "followup_question": str(data.get("followup_question", "")).strip(),
        }

    def decompose_user_query(self, question: str) -> List[str]:
        question = question.strip()
        if not question:
            return []

        target_language = self._question_language(question)
        language_instruction = (
            "The user question is in English. Output every query in English."
            if target_language == "en"
            else "用户问题是中文。请用中文输出每个 query。"
            if target_language == "zh"
            else "每个 query 使用和用户问题一致的语言。"
        )
        prompt = f"""
你是主 Agent 的 query 拆解器。请判断【用户单轮问题】是否包含多个可以独立处理的诉求，并拆成若干条可独立路由给子 Agent 的 query。
只输出 JSON 字符串数组，不要输出 Markdown、解释文字或代码块。

拆解规则：
1. 如果只有一个诉求，输出只包含原问题的数组。
2. 如果同时包含客服政策/售后/订单/退款/发票/投诉等问题，以及产品手册/操作步骤/清洁/安装/故障排除等问题，必须拆开。
3. 如果同一句里有多个互不依赖的问题，也要拆开。
4. 不要添加用户没有问的新问题。
5. 每个 query 必须保留必要对象、产品名、型号、功能名、故障现象和上下文。
6. {language_instruction}

示例：
输入：请问你们能开发票吗？另外空气净化器滤网怎么更换？
输出：["请问你们能开发票吗？", "空气净化器滤网怎么更换？"]

输入：I want to know your refund policy and how to clean the vacuum cleaner filter.
输出：["I want to know your refund policy.", "How do I clean the vacuum cleaner filter?"]

输入：如何清洁空调空气滤网？
输出：["如何清洁空调空气滤网？"]

【用户单轮问题】
{question}

请只输出 JSON 数组：
""".strip()

        resp = self.client.chat.completions.create(
            model=self.settings.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": "Decompose one user query into independently routable sub-queries. Output only a JSON array of strings.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        decomposed = self._parse_rewrite_json(resp.choices[0].message.content)
        decomposed = [item.strip() for item in decomposed if item.strip()]
        if not decomposed:
            return [question]
        return self._dedupe_keep_order(decomposed)

    @staticmethod
    def _parse_rewrite_json(content: str) -> List[str]:
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return []
        if not isinstance(data, list) or not all(
            isinstance(item, str) for item in data
        ):
            return []
        return data

    @classmethod
    def _parse_current_turn_rewrite(cls, content: str) -> Dict[str, Any] | None:
        text = cls._strip_json_fence(content)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = cls._load_first_json_object(text)
        if not isinstance(data, dict):
            return None
        return data

    @classmethod
    def _parse_route_plan(cls, content: str) -> Dict[str, Any] | None:
        text = cls._strip_json_fence(content)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = cls._load_first_json_object(text)
        if not isinstance(data, dict):
            return None

        intent = data.get("intent")
        if intent not in {"customer_service", "manual"}:
            return None

        confidence = data.get("confidence")
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            confidence_value = 0.0
        confidence_value = max(0.0, min(1.0, confidence_value))

        reason = data.get("reason")
        if not isinstance(reason, str):
            reason = ""

        return {
            "intent": intent,
            "confidence": confidence_value,
            "reason": reason.strip(),
        }

    @classmethod
    def _parse_customer_analysis(cls, content: str, question: str) -> Dict[str, Any]:
        text = cls._strip_json_fence(content)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = cls._load_first_json_object(text)
        if not isinstance(data, dict):
            return cls._fallback_customer_analysis(question, "analysis_json_parse_failed")

        main_intent = data.get("main_intent")
        if not isinstance(main_intent, str) or not main_intent.strip():
            main_intent = question

        sub_questions = data.get("sub_questions")
        if not isinstance(sub_questions, list):
            sub_questions = [question]
        sub_questions = [
            str(item).strip() for item in sub_questions if str(item).strip()
        ]
        if not sub_questions:
            sub_questions = [question]

        required_info = data.get("required_info")
        if not isinstance(required_info, list):
            required_info = []
        required_info = [
            str(item).strip() for item in required_info if str(item).strip()
        ]

        risk_notes = data.get("risk_notes")
        if not isinstance(risk_notes, list):
            risk_notes = []
        risk_notes = [
            str(item).strip() for item in risk_notes if str(item).strip()
        ]

        return {
            "main_intent": main_intent.strip(),
            "sub_questions": cls._dedupe_keep_order(sub_questions),
            "required_info": cls._dedupe_keep_order(required_info),
            "risk_notes": cls._dedupe_keep_order(risk_notes),
        }

    @classmethod
    def _parse_retrieval_plan(cls, content: str, question: str) -> Dict[str, Any]:
        text = cls._strip_json_fence(content)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = cls._load_first_json_object(text)
        if not isinstance(data, dict):
            return cls._fallback_retrieval_plan(question, "planner_json_parse_failed")

        sub_questions = data.get("sub_questions")
        if not isinstance(sub_questions, list):
            sub_questions = [question]
        sub_questions = [str(item).strip() for item in sub_questions if str(item).strip()]
        if not sub_questions:
            sub_questions = [question]

        evidence_sufficient = data.get("evidence_sufficient")
        if not isinstance(evidence_sufficient, bool):
            evidence_sufficient = False

        search_queries = data.get("search_queries")
        if not isinstance(search_queries, list):
            search_queries = []
        search_queries = [str(item).strip() for item in search_queries if str(item).strip()]
        search_queries = cls._dedupe_keep_order(search_queries)[:3]
        if evidence_sufficient:
            search_queries = []

        reason = data.get("reason")
        if not isinstance(reason, str):
            reason = ""

        return {
            "sub_questions": sub_questions,
            "evidence_sufficient": evidence_sufficient,
            "search_queries": search_queries,
            "reason": reason.strip(),
        }

    @classmethod
    def _parse_evidence_selection(
        cls,
        content: str,
        candidate_chunks: List[RetrievedChunk],
        max_chunks: int,
    ) -> Dict[str, Any]:
        text = cls._strip_json_fence(content)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = cls._load_first_json_object(text)
        if not isinstance(data, dict):
            return cls._fallback_evidence_selection(
                candidate_chunks, max_chunks, "selector_json_parse_failed"
            )

        valid_ids = {chunk.chunk_id for chunk in candidate_chunks}
        selected = data.get("selected_chunk_ids")
        if not isinstance(selected, list):
            selected = []
        selected_ids = []
        for item in selected:
            chunk_id = str(item).strip()
            if chunk_id in valid_ids and chunk_id not in selected_ids:
                selected_ids.append(chunk_id)
            if len(selected_ids) >= max_chunks:
                break

        if not selected_ids:
            return cls._fallback_evidence_selection(
                candidate_chunks, max_chunks, "selector_no_valid_chunk_ids"
            )

        coverage = data.get("coverage")
        if not isinstance(coverage, list):
            coverage = []

        reason = data.get("reason")
        if not isinstance(reason, str):
            reason = ""

        return {
            "selected_chunk_ids": selected_ids,
            "coverage": coverage,
            "reason": reason.strip(),
            "fallback": False,
        }

    @staticmethod
    def _fallback_evidence_selection(
        candidate_chunks: List[RetrievedChunk], max_chunks: int, reason: str
    ) -> Dict[str, Any]:
        return {
            "selected_chunk_ids": [
                chunk.chunk_id for chunk in candidate_chunks[:max_chunks]
            ],
            "coverage": [],
            "reason": reason,
            "fallback": True,
        }

    @staticmethod
    def _strip_json_fence(content: str) -> str:
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        return text.strip()

    @staticmethod
    def _load_first_json_object(text: str) -> Dict[str, Any] | None:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _fallback_retrieval_plan(question: str, reason: str) -> Dict[str, Any]:
        return {
            "sub_questions": [question],
            "evidence_sufficient": True,
            "search_queries": [],
            "reason": reason,
        }

    @staticmethod
    def _fallback_customer_analysis(question: str, reason: str) -> Dict[str, Any]:
        return {
            "main_intent": question,
            "sub_questions": [question],
            "required_info": [],
            "risk_notes": [reason],
        }

    @staticmethod
    def _dedupe_keep_order(items: List[str]) -> List[str]:
        seen = set()
        result = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            result.append(item)
        return result

    @staticmethod
    def _question_language(text: str) -> str:
        cjk = len(CJK_RE.findall(text))
        letters = len(ASCII_LETTER_RE.findall(text))
        if cjk > 0:
            return "zh"
        if letters > 0:
            return "en"
        return "unknown"

    @staticmethod
    def _build_selection_context(chunks: List[RetrievedChunk]) -> str:
        blocks = []
        for i, c in enumerate(chunks, start=1):
            blocks.append(
                f"---\n"
                f"candidate_index: {i}\n"
                f"chunk_id: {c.chunk_id}\n"
                f"manual_id: {c.manual_id}\n"
                f"section_path: {c.section_path}\n"
                f"rerank_score: {c.rerank_score}\n"
                f"score_source: {c.score_source}\n"
                f"image_ids: {', '.join(c.image_ids) if c.image_ids else '无'}\n"
                f"text:\n{c.text}"
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _fix_customer_rewrites(
        original_turns: List[str], rewritten_turns: List[str]
    ) -> List[str]:
        fixed = list(rewritten_turns)
        for idx in range(1, len(fixed)):
            prev = original_turns[idx - 1]
            cur = original_turns[idx]
            rewritten = fixed[idx]
            if "发票" in prev and "收到" in cur and "发票" not in rewritten:
                fixed[idx] = "发票多久能收到？"
            elif "7天无理由" in prev and "条件" in cur and "7天无理由" not in rewritten:
                fixed[idx] = "7天无理由退货需要满足什么条件？"
            elif (
                "发票" in prev
                and "抬头" in prev
                and "重新开具" in cur
                and "发票" not in rewritten
            ):
                fixed[idx] = "发票抬头写错了，能重新开具吗？"
        return fixed

    @staticmethod
    def _build_context(chunks: List[RetrievedChunk]) -> str:
        blocks = []
        for i, c in enumerate(chunks, start=1):
            blocks.append(
                f"---\n"
                f"manual_id: {c.manual_id}\n"
                f"section_path: {c.section_path}\n"
                f"image_ids: {', '.join(c.image_ids) if c.image_ids else '无'}\n"
                f"text:\n{c.text}"
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _clean_answer(answer: str) -> str:
        text = EMOJI_RE.sub("", answer)
        text = MARKDOWN_BOLD_RE.sub(r"\1", text)
        text = EVIDENCE_REF_RE.sub("", text)
        return text.strip()

    def _ensure_question_language(self, question: str, answer: str) -> str:
        if self._looks_english(question) and CJK_RE.search(answer):
            resp = self.client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "Translate the answer into natural English. Preserve <PIC> tokens exactly. Do not add new content.",
                    },
                    {"role": "user", "content": answer},
                ],
                temperature=0.0,
            )
            return self._clean_answer(resp.choices[0].message.content)
        return answer

    @staticmethod
    def _looks_english(text: str) -> bool:
        letters = len(ASCII_LETTER_RE.findall(text))
        cjk = len(CJK_RE.findall(text))
        return letters >= 10 and letters > cjk * 3
