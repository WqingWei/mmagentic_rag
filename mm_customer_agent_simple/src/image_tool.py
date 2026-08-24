import base64
import binascii
import json
import mimetypes
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from openai import OpenAI

from .config import Settings


DATA_URL_RE = re.compile(r"^data:(image/[^;]+);base64,", re.I)


class ImageUnderstandingTool:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._reader = None
        self.api_vlm_tool = ApiVLMImageTool(settings)
        self.local_vlm_tool = LocalVLMImageTool(settings)

    def observe(
        self,
        image_paths: List[str] | None = None,
        image_base64: List[str] | None = None,
        user_question: str = "",
    ) -> Dict[str, Any]:
        inputs = self._build_image_inputs(image_paths or [], image_base64 or [])
        if not inputs:
            return {}
        if len(inputs) > self.settings.max_image_inputs:
            raise ValueError(f"too many images: max {self.settings.max_image_inputs}")

        tools = self._enabled_tools()
        images = []
        if "ocr" in tools:
            for idx, item in enumerate(inputs, start=1):
                images.append(self._inspect_image(idx, item))
        else:
            for idx, item in enumerate(inputs, start=1):
                images.append(self._basic_image_info(idx, item))

        ocr_text = self._combined_ocr_text(images)
        vlm_result: Dict[str, Any] = {}
        if "api_vlm" in tools:
            vlm_result = self.api_vlm_tool.observe(inputs, user_question, ocr_text)
        if "local_vlm" in tools or (
            self.settings.local_vlm_enabled and not vlm_result
        ):
            vlm_result = self.local_vlm_tool.observe(inputs, user_question, ocr_text)

        images = self._merge_image_vlm(images, vlm_result.get("images", []))
        likely_intent = str(vlm_result.get("likely_intent", "unknown") or "unknown").strip()
        if likely_intent == "image_direct":
            likely_intent = "unknown"
        risk_notes = [
            "OCR may miss small or blurry text.",
            "Image observations are clues for query rewriting and routing, not final manual or policy evidence.",
        ]
        risk_notes.extend(
            str(item) for item in vlm_result.get("risk_notes", []) if str(item).strip()
        )

        return {
            "image_count": len(inputs),
            "max_image_inputs": self.settings.max_image_inputs,
            "tools": tools,
            "ocr_text": ocr_text,
            "summary": (
                f"OCR and VLM observations were extracted from {len(inputs)} image(s)."
                if vlm_result
                else f"OCR text was extracted from {len(inputs)} image(s)."
            ),
            "vlm_summary": str(vlm_result.get("summary", "")).strip(),
            "product_type": str(vlm_result.get("product_type", "")).strip(),
            "brand_or_model": str(vlm_result.get("brand_or_model", "")).strip(),
            "visible_parts": self._as_list(vlm_result.get("visible_parts")),
            "visible_error_code": str(vlm_result.get("visible_error_code", "")).strip(),
            "likely_intent": likely_intent,
            "suggested_query": self._suggested_query(user_question, ocr_text, vlm_result),
            "images": images,
            "risk_notes": self._dedupe(risk_notes),
        }

    def build_augmented_question(self, question: str, observation: Dict[str, Any]) -> str:
        if not observation:
            return question
        context = self.compact_context(observation)
        if context:
            return f"{question.strip()}\n\nImage context:\n{context}".strip()

        fields = [
            ("summary", "image_summary"),
            ("ocr_text", "image_ocr_text"),
            ("likely_intent", "image_likely_intent"),
            ("suggested_query", "image_suggested_query"),
        ]
        lines = []
        for source_key, label in fields:
            value = observation.get(source_key)
            if isinstance(value, str) and value.strip():
                lines.append(f"{label}: {value.strip()}")

        images = observation.get("images")
        if isinstance(images, list) and images:
            for image in images:
                if not isinstance(image, dict):
                    continue
                index = image.get("index", "")
                filename = image.get("filename", "")
                width = image.get("width", "")
                height = image.get("height", "")
                image_ocr = str(image.get("ocr_text", "")).strip()
                lines.append(
                    f"image_{index}: filename={filename} size={width}x{height}"
                )
                if image_ocr:
                    lines.append(f"image_{index}_ocr_text: {image_ocr}")

        risk_notes = observation.get("risk_notes")
        if isinstance(risk_notes, list) and risk_notes:
            lines.append(
                "image_risk_notes: "
                + "; ".join(str(item).strip() for item in risk_notes if str(item).strip())
            )

        if not lines:
            return question

        image_context = "\n".join(lines)
        return (
            f"{question.strip()}\n\n"
            "The user also provided image input. Use this image observation as context, "
            "but answer only with information supported by the image and the knowledge base:\n"
            f"{image_context}"
        ).strip()

    @classmethod
    def compact_context(cls, observation: Dict[str, Any]) -> str:
        lines: List[str] = []
        for key, label in [
            ("ocr_text", "ocr"),
            ("vlm_summary", "visual_summary"),
            ("product_type", "product_type"),
            ("brand_or_model", "brand_or_model"),
            ("visible_error_code", "visible_error_code"),
            ("likely_intent", "likely_intent"),
            ("suggested_query", "suggested_query"),
        ]:
            value = observation.get(key)
            if isinstance(value, str) and value.strip():
                lines.append(f"{label}: {value.strip()}")
        visible_parts = observation.get("visible_parts")
        if isinstance(visible_parts, list) and visible_parts:
            lines.append("visible_parts: " + ", ".join(str(item) for item in visible_parts))
        for image in observation.get("images", []) or []:
            if not isinstance(image, dict):
                continue
            parts = [f"image_{image.get('index')}"]
            if image.get("filename"):
                parts.append(f"file={image.get('filename')}")
            if image.get("width") and image.get("height"):
                parts.append(f"size={image.get('width')}x{image.get('height')}")
            if str(image.get("ocr_text", "")).strip():
                parts.append(f"ocr={str(image.get('ocr_text')).strip()}")
            if str(image.get("vlm_summary", "")).strip():
                parts.append(f"vlm={str(image.get('vlm_summary')).strip()}")
            lines.append("; ".join(parts))
        return "\n".join(lines).strip()

    def _build_image_inputs(
        self,
        image_paths: List[str],
        image_base64: List[str],
    ) -> List[Dict[str, Any]]:
        inputs: List[Dict[str, Any]] = []
        for item in image_base64:
            value = str(item).strip()
            if not value:
                continue
            inputs.append(self._base64_input(value))
        for item in image_paths:
            value = str(item).strip()
            if not value:
                continue
            inputs.append(self._path_input(value))
        return inputs

    def _enabled_tools(self) -> List[str]:
        tools = [
            item.strip().lower()
            for item in self.settings.image_analysis_tools.split(",")
            if item.strip()
        ]
        if self.settings.local_vlm_enabled and "local_vlm" not in tools:
            tools.append("local_vlm")
        return tools or ["ocr"]

    @staticmethod
    def _path_input(raw_path: str) -> Dict[str, Any]:
        path = Path(raw_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"image file not found: {raw_path}")
        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        image_bytes = path.read_bytes()
        return {
            "source": "path",
            "filename": path.name,
            "mime_type": mime_type,
            "size_bytes": path.stat().st_size,
            "bytes": image_bytes,
            "data_url": _data_url(mime_type, image_bytes),
        }

    @staticmethod
    def _base64_input(value: str) -> Dict[str, Any]:
        match = DATA_URL_RE.match(value)
        mime_type = match.group(1).lower() if match else "image/png"
        raw_base64 = DATA_URL_RE.sub("", value, count=1) if match else value
        try:
            image_bytes = base64.b64decode(raw_base64, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("invalid image_base64 payload") from exc
        return {
            "source": "base64",
            "filename": "",
            "mime_type": mime_type,
            "size_bytes": len(image_bytes),
            "bytes": image_bytes,
            "data_url": _data_url(mime_type, image_bytes),
        }

    def _inspect_image(self, index: int, item: Dict[str, Any]) -> Dict[str, Any]:
        image = self._load_pil_image(item["bytes"])
        width, height = image.size
        ocr_items = self._read_ocr(image)
        ocr_text = "\n".join(item["text"] for item in ocr_items).strip()
        return {
            "index": index,
            "source": item["source"],
            "filename": item["filename"],
            "mime_type": item["mime_type"],
            "width": width,
            "height": height,
            "size_bytes": item["size_bytes"],
            "ocr_text": ocr_text,
            "ocr_items": ocr_items,
        }

    def _basic_image_info(self, index: int, item: Dict[str, Any]) -> Dict[str, Any]:
        image = self._load_pil_image(item["bytes"])
        width, height = image.size
        return {
            "index": index,
            "source": item["source"],
            "filename": item["filename"],
            "mime_type": item["mime_type"],
            "width": width,
            "height": height,
            "size_bytes": item["size_bytes"],
            "ocr_text": "",
            "ocr_items": [],
        }

    @staticmethod
    def _combined_ocr_text(images: List[Dict[str, Any]]) -> str:
        parts = []
        for image in images:
            text = str(image.get("ocr_text", "")).strip()
            if text:
                parts.append(f"[image_{image.get('index')}] {text}")
        return "\n".join(parts).strip()

    @classmethod
    def _suggested_query(
        cls,
        user_question: str,
        ocr_text: str,
        vlm_result: Dict[str, Any],
    ) -> str:
        parts = [user_question.strip()]
        for key in ["suggested_query", "summary", "visible_error_code", "product_type"]:
            value = str(vlm_result.get(key, "")).strip()
            if value:
                parts.append(value)
        if ocr_text:
            parts.append(f"OCR: {ocr_text}")
        return "\n".join(cls._dedupe([item for item in parts if item])).strip()

    @staticmethod
    def _merge_image_vlm(
        images: List[Dict[str, Any]],
        vlm_images: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        by_index = {
            int(item.get("index") or idx): item
            for idx, item in enumerate(vlm_images, start=1)
            if isinstance(item, dict)
        }
        merged = []
        for image in images:
            item = dict(image)
            vlm_item = by_index.get(int(item.get("index") or 0), {})
            if vlm_item:
                item["vlm_summary"] = str(vlm_item.get("vlm_summary") or vlm_item.get("summary") or "").strip()
                item["vlm_observation"] = vlm_item.get("vlm_observation", vlm_item)
            merged.append(item)
        return merged

    @staticmethod
    def _as_list(value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    @staticmethod
    def _dedupe(items: List[str]) -> List[str]:
        result: List[str] = []
        seen = set()
        for item in items:
            value = str(item).strip()
            if not value or value in seen:
                continue
            seen.add(value)
            result.append(value)
        return result

    @staticmethod
    def _load_pil_image(image_bytes: bytes):
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required for local OCR image loading") from exc

        try:
            image = Image.open(BytesIO(image_bytes))
            return image.convert("RGB")
        except Exception as exc:
            raise ValueError("invalid or unsupported image file") from exc

    def _read_ocr(self, image) -> List[Dict[str, Any]]:
        reader = self._get_reader()
        raw_results = reader.readtext(np.array(image), detail=1)
        items: List[Dict[str, Any]] = []
        for result in raw_results:
            if len(result) < 3:
                continue
            text = str(result[1]).strip()
            if not text:
                continue
            try:
                confidence = float(result[2])
            except (TypeError, ValueError):
                confidence = 0.0
            items.append({"text": text, "confidence": confidence})
        return items

    def _get_reader(self):
        if self._reader is not None:
            return self._reader
        try:
            import easyocr
        except ImportError as exc:
            raise RuntimeError(
                "EasyOCR is required for local OCR. Install it with: pip install easyocr"
            ) from exc
        languages = [
            item.strip()
            for item in self.settings.ocr_languages.split(",")
            if item.strip()
        ] or ["ch_sim", "en"]
        model_storage_dir = Path(self.settings.ocr_model_storage_dir)
        model_storage_dir.mkdir(parents=True, exist_ok=True)
        user_network_dir = model_storage_dir / "user_network"
        user_network_dir.mkdir(parents=True, exist_ok=True)
        self._reader = easyocr.Reader(
            languages,
            gpu=False,
            model_storage_directory=str(model_storage_dir),
            user_network_directory=str(user_network_dir),
        )
        return self._reader


class ApiVLMImageTool:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = OpenAI(
            api_key=settings.dashscope_api_key,
            base_url=settings.api_vlm_base_url,
        )

    def observe(
        self,
        inputs: List[Dict[str, Any]],
        user_question: str,
        ocr_text: str,
    ) -> Dict[str, Any]:
        return _call_vlm(
            self.client,
            self.settings.api_vlm_model,
            inputs,
            user_question,
            ocr_text,
            provider_name="api_vlm",
        )


class LocalVLMImageTool:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = OpenAI(
            api_key=settings.dashscope_api_key or "local-vlm",
            base_url=settings.local_vlm_base_url,
        )

    def observe(
        self,
        inputs: List[Dict[str, Any]],
        user_question: str,
        ocr_text: str,
    ) -> Dict[str, Any]:
        model = self.settings.local_vlm_model.strip()
        if not model:
            raise RuntimeError("LOCAL_VLM_MODEL must be set when local VLM is enabled")
        return _call_vlm(
            self.client,
            model,
            inputs,
            user_question,
            ocr_text,
            provider_name="local_vlm",
        )


def _call_vlm(
    client: OpenAI,
    model: str,
    inputs: List[Dict[str, Any]],
    user_question: str,
    ocr_text: str,
    provider_name: str,
) -> Dict[str, Any]:
    if not model:
        raise RuntimeError(f"{provider_name} model is not configured")
    prompt = f"""
Analyze the uploaded image(s) for a customer-support routing and query-rewriting agent.
Return only JSON with these fields:
summary, product_type, brand_or_model, visible_parts, visible_error_code,
likely_intent, suggested_query, risk_notes, images.
likely_intent must be one of: manual, customer_service, unknown.
Describe visible facts only. Do not invent product model, error meaning, repair steps, or service-policy conclusions.

User question:
{user_question}

OCR text:
{ocr_text}
""".strip()
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for item in inputs:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": item["data_url"]},
            }
        )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a precise visual observation tool. Output only valid JSON.",
            },
            {"role": "user", "content": content},
        ],
        temperature=0.0,
    )
    text = resp.choices[0].message.content or "{}"
    data = _parse_json_object(text)
    images = data.get("images")
    if not isinstance(images, list):
        images = [
            {
                "index": idx,
                "vlm_summary": str(data.get("summary", "")).strip(),
                "vlm_observation": {},
            }
            for idx in range(1, len(inputs) + 1)
        ]
    data["images"] = images
    data["provider"] = provider_name
    return data


def _parse_json_object(text: str) -> Dict[str, Any]:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    try:
        data = json.loads(value)
    except json.JSONDecodeError as exc:
        match = re.search(r"\{.*\}", value, re.S)
        if not match:
            raise RuntimeError(f"VLM returned invalid JSON: {text[:200]}") from exc
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise RuntimeError("VLM returned JSON that is not an object")
    return data


def _data_url(mime_type: str, image_bytes: bytes) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
