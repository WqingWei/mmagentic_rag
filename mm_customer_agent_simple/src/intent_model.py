from __future__ import annotations

import asyncio
import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Protocol

from .config import Settings


class IntentType(str, Enum):
    KNOWLEDGE = "knowledge"
    CUSTOMER_SERVICE = "customer_service"
    CLARIFICATION = "clarification"
    MIXED = "mixed"


@dataclass(frozen=True)
class IntentPrediction:
    intent: IntentType
    confidence: float
    reason: str
    secondary_intents: List[IntentType] = field(default_factory=list)
    needs_planner: bool = False
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["intent"] = self.intent.value
        data["secondary_intents"] = [item.value for item in self.secondary_intents]
        return data


class IntentClassifier(Protocol):
    async def classify(
        self,
        query: str,
        history: List[Dict[str, Any]] | None = None,
        has_images: bool = False,
    ) -> IntentPrediction: ...


KNOWLEDGE_TERMS = (
    "如何",
    "怎么",
    "怎样",
    "步骤",
    "使用",
    "操作",
    "安装",
    "设置",
    "连接",
    "清洁",
    "更换",
    "故障",
    "报错",
    "说明书",
    "manual",
    "how to",
    "troubleshoot",
    "install",
    "replace",
    "clean",
)
CUSTOMER_TERMS = (
    "订单",
    "物流",
    "快递",
    "退款",
    "退货",
    "换货",
    "发票",
    "投诉",
    "售后",
    "赔偿",
    "发货",
    "order",
    "shipping",
    "refund",
    "return",
    "invoice",
    "complaint",
    "after-sales",
)
AMBIGUOUS_TERMS = (
    "怎么办",
    "怎么弄",
    "帮我看看",
    "这个呢",
    "有问题",
    "不行了",
    "处理一下",
    "what now",
    "help me",
    "it does not work",
)


class RuleIntentClassifier:
    """Deterministic fallback and baseline for four-way intent routing."""

    def __init__(self, confidence_threshold: float = 0.65):
        self.confidence_threshold = confidence_threshold

    async def classify(
        self,
        query: str,
        history: List[Dict[str, Any]] | None = None,
        has_images: bool = False,
    ) -> IntentPrediction:
        text = query.strip().lower()
        has_knowledge = any(term in text for term in KNOWLEDGE_TERMS)
        has_customer = any(term in text for term in CUSTOMER_TERMS)
        has_history = bool(history)
        complex_query = self._is_complex(query)

        if has_knowledge and has_customer:
            intent = IntentType.MIXED
            confidence = 0.92
            reason = "同时包含产品知识与客服业务诉求"
            secondary = [IntentType.KNOWLEDGE, IntentType.CUSTOMER_SERVICE]
        elif has_customer:
            intent = IntentType.CUSTOMER_SERVICE
            confidence = 0.86
            reason = "包含订单、交易或售后服务信号"
            secondary = []
        elif has_knowledge:
            intent = IntentType.KNOWLEDGE
            confidence = 0.84
            reason = "包含产品使用、说明书或故障排查信号"
            secondary = []
        elif self._is_ambiguous(text, has_images, has_history):
            intent = IntentType.CLARIFICATION
            confidence = 0.78
            reason = "缺少可执行的对象、现象或目标"
            secondary = []
        else:
            intent = IntentType.CLARIFICATION
            confidence = 0.55
            reason = "未发现稳定的知识或客服意图信号"
            secondary = []

        return IntentPrediction(
            intent=intent,
            confidence=confidence,
            reason=reason,
            secondary_intents=secondary,
            needs_planner=(
                intent == IntentType.MIXED
                or has_images
                or has_history
                or complex_query
                or confidence < self.confidence_threshold
            ),
            source="rule",
        )

    @staticmethod
    def _is_ambiguous(text: str, has_images: bool, has_history: bool) -> bool:
        if has_images or has_history:
            return False
        compact = re.sub(r"[\s，。！？,.!?]", "", text)
        return len(compact) <= 5 or any(term in text for term in AMBIGUOUS_TERMS)

    @staticmethod
    def _is_complex(query: str) -> bool:
        chinese_separators = ("另外", "同时", "并且", "还有", "以及", "然后")
        english_complex = re.search(r"\b(?:also|and)\b", query.lower()) is not None
        return (
            "\n" in query
            or sum(query.count(mark) for mark in "？?；;") > 1
            or any(term in query for term in chinese_separators)
            or english_complex
        )


INTENT_TEACHER_PROMPT = """
你是多模态客服系统的意图教师模型。根据最近对话、当前问题和是否带图片，执行四分类。

标签定义：
- knowledge：产品知识、说明书、操作、安装、设置、清洁、参数、故障排查。
- customer_service：订单、物流、退款、退换货、发票、投诉、赔偿和售后流程。
- clarification：信息不足、指代无法消解、目标不明确，需要先追问。
- mixed：同一请求包含两个或以上可独立执行的 knowledge/customer_service 诉求。

只输出 JSON：
{{
  "intent": "knowledge|customer_service|clarification|mixed",
  "confidence": 0.0,
  "reason": "不超过60字",
  "secondary_intents": ["knowledge", "customer_service"],
  "needs_planner": true
}}

needs_planner=true 的条件：混合意图、多轮指代、复杂多诉求、带图片，或置信度不足。

最近对话：
{history}

当前问题：
{query}

当前请求带图片：{has_images}
""".strip()


class PromptIntentClassifier:
    """Teacher/API classifier implemented with AgentScope model abstraction."""

    def __init__(
        self,
        settings: Settings,
        framework: Any,
        fallback: IntentClassifier | None = None,
        local: bool = False,
    ):
        self.settings = settings
        self.framework = framework
        self.fallback = fallback or RuleIntentClassifier(
            settings.intent_confidence_threshold
        )
        self.local = local

    async def classify(
        self,
        query: str,
        history: List[Dict[str, Any]] | None = None,
        has_images: bool = False,
    ) -> IntentPrediction:
        history = history or []
        try:
            model = self._build_model()
            response = await model(
                messages=[
                    {
                        "role": "system",
                        "content": "Classify customer intent. Output JSON only.",
                    },
                    {
                        "role": "user",
                        "content": INTENT_TEACHER_PROMPT.format(
                            history=json.dumps(history, ensure_ascii=False),
                            query=query,
                            has_images=str(has_images).lower(),
                        ),
                    },
                ]
            )
            data = self._parse_json(self._response_text(response))
            prediction = self._prediction_from_dict(data, history, has_images)
            return IntentPrediction(
                **{**prediction.__dict__, "source": "local_llm" if self.local else "prompt_api"}
            )
        except Exception:
            if not self.settings.intent_fallback_to_rules:
                raise
            return await self.fallback.classify(query, history, has_images)

    def _build_model(self) -> Any:
        imports = self.framework._load_agentscope()
        if self.local:
            model_name = self.settings.intent_local_model
            base_url = self.settings.intent_local_base_url
            api_key = self.settings.intent_local_api_key or "local"
        else:
            model_name = self.settings.intent_api_model
            base_url = self.settings.intent_api_base_url
            api_key = self.settings.intent_api_key
        return imports["OpenAIChatModel"](
            model_name=model_name,
            api_key=api_key,
            stream=False,
            client_kwargs={"base_url": base_url},
            generate_kwargs={"temperature": 0.0},
        )

    def _prediction_from_dict(
        self,
        data: Dict[str, Any],
        history: List[Dict[str, Any]],
        has_images: bool,
    ) -> IntentPrediction:
        intent = IntentType(str(data.get("intent", "clarification")))
        confidence = min(max(float(data.get("confidence", 0.0)), 0.0), 1.0)
        secondary = []
        for item in data.get("secondary_intents", []) or []:
            try:
                secondary.append(IntentType(str(item)))
            except ValueError:
                continue
        return IntentPrediction(
            intent=intent,
            confidence=confidence,
            reason=str(data.get("reason", "")).strip(),
            secondary_intents=secondary,
            needs_planner=bool(data.get("needs_planner"))
            or intent == IntentType.MIXED
            or bool(history)
            or has_images
            or confidence < self.settings.intent_confidence_threshold,
            source="",
        )

    @staticmethod
    def _response_text(response: Any) -> str:
        texts = []
        for block in getattr(response, "content", []) or []:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(str(block.get("text", "")))
            elif getattr(block, "type", "") == "text":
                texts.append(str(getattr(block, "text", "")))
        return "".join(texts).strip()

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end < start:
            raise ValueError("intent model did not return JSON")
        value = json.loads(cleaned[start : end + 1])
        if not isinstance(value, dict):
            raise ValueError("intent JSON must be an object")
        return value


class BertIntentClassifier:
    """Optional distilled BERT four-class classifier loaded lazily."""

    def __init__(self, settings: Settings, fallback: IntentClassifier | None = None):
        self.settings = settings
        self.fallback = fallback or RuleIntentClassifier(
            settings.intent_confidence_threshold
        )
        self._tokenizer: Any = None
        self._model: Any = None

    async def classify(
        self,
        query: str,
        history: List[Dict[str, Any]] | None = None,
        has_images: bool = False,
    ) -> IntentPrediction:
        try:
            return await asyncio.to_thread(
                self._classify_sync,
                query,
                history or [],
                has_images,
            )
        except Exception:
            if not self.settings.intent_fallback_to_rules:
                raise
            return await self.fallback.classify(query, history, has_images)

    def _classify_sync(
        self,
        query: str,
        history: List[Dict[str, Any]],
        has_images: bool,
    ) -> IntentPrediction:
        self._ensure_model()
        import torch

        history_text = "\n".join(
            f"{item.get('role', '')}: {item.get('content', '')}" for item in history[-6:]
        )
        text = f"[HISTORY]\n{history_text}\n[QUERY]\n{query}\n[HAS_IMAGE]\n{has_images}"
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.settings.intent_bert_max_length,
        )
        with torch.no_grad():
            probabilities = torch.softmax(self._model(**inputs).logits, dim=-1)[0]
        index = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[index].item())
        raw_label = self._model.config.id2label.get(index, str(index)).lower()
        intent = self._normalize_bert_label(raw_label, index)
        return IntentPrediction(
            intent=intent,
            confidence=confidence,
            reason="distilled_bert_prediction",
            secondary_intents=[],
            needs_planner=(
                intent == IntentType.MIXED
                or bool(history)
                or has_images
                or confidence < self.settings.intent_confidence_threshold
            ),
            source="bert",
        )

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        model_path = Path(self.settings.intent_bert_model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"intent BERT model not found: {model_path}")
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self._model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        self._model.eval()

    @staticmethod
    def _normalize_bert_label(label: str, index: int) -> IntentType:
        aliases = {
            "knowledge": IntentType.KNOWLEDGE,
            "customer_service": IntentType.CUSTOMER_SERVICE,
            "customer": IntentType.CUSTOMER_SERVICE,
            "clarification": IntentType.CLARIFICATION,
            "mixed": IntentType.MIXED,
        }
        if label in aliases:
            return aliases[label]
        labels = list(IntentType)
        if 0 <= index < len(labels):
            return labels[index]
        raise ValueError(f"unsupported BERT intent label: {label!r} (index={index})")


def build_intent_classifier(settings: Settings, framework: Any) -> IntentClassifier:
    backend = settings.intent_model_backend.strip().lower()
    rule = RuleIntentClassifier(settings.intent_confidence_threshold)
    if backend == "bert":
        return BertIntentClassifier(settings, fallback=rule)
    if backend in {"local_llm", "0.6b", "small_llm"}:
        return PromptIntentClassifier(settings, framework, fallback=rule, local=True)
    if backend in {"prompt_api", "api", "llm"}:
        return PromptIntentClassifier(settings, framework, fallback=rule, local=False)
    if backend == "rule":
        return rule
    raise ValueError(
        "INTENT_MODEL_BACKEND must be one of: "
        "bert, local_llm (or 0.6b), prompt_api, rule"
    )
