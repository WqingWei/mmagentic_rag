import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent

load_dotenv(WORKSPACE_ROOT / ".env")
load_dotenv(PACKAGE_ROOT / ".env", override=False)


def _resolve_path(raw_path: str, fallback_base: Path = PACKAGE_ROOT) -> str:
    path = Path(raw_path)
    if path.is_absolute():
        return str(path)

    candidates = [
        Path.cwd() / path,
        PACKAGE_ROOT / path,
        WORKSPACE_ROOT / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str((fallback_base / path).resolve())


def _resolve_storage_dir(raw_path: str) -> str:
    path = Path(raw_path)
    if path.is_absolute():
        return str(path)

    candidates = [
        Path.cwd() / path,
        PACKAGE_ROOT / path,
        WORKSPACE_ROOT / path,
    ]
    for candidate in candidates:
        if (candidate / "bm25.pkl").exists():
            return str(candidate.resolve())
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str((PACKAGE_ROOT / path).resolve())


RAW_STORAGE_DIR = os.getenv("STORAGE_DIR", "storage")
DEFAULT_STORAGE_DIR = _resolve_storage_dir(RAW_STORAGE_DIR)
RAW_QDRANT_PATH = os.getenv("QDRANT_PATH")
DEFAULT_QDRANT_PATH = (
    _resolve_path(RAW_QDRANT_PATH)
    if RAW_QDRANT_PATH
    else str((Path(DEFAULT_STORAGE_DIR) / "qdrant").resolve())
)
RAW_CUSTOMER_QA_QDRANT_PATH = os.getenv("CUSTOMER_QA_QDRANT_PATH")
DEFAULT_CUSTOMER_QA_QDRANT_PATH = (
    _resolve_path(RAW_CUSTOMER_QA_QDRANT_PATH)
    if RAW_CUSTOMER_QA_QDRANT_PATH
    else str((Path(DEFAULT_STORAGE_DIR) / "customer_qa_qdrant").resolve())
)
RAW_OCR_MODEL_STORAGE_DIR = os.getenv("OCR_MODEL_STORAGE_DIR")
DEFAULT_OCR_MODEL_STORAGE_DIR = (
    _resolve_path(RAW_OCR_MODEL_STORAGE_DIR)
    if RAW_OCR_MODEL_STORAGE_DIR
    else str((Path(DEFAULT_STORAGE_DIR) / "easyocr_models").resolve())
)


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def _get_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    dashscope_api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    dashscope_base_url: str = os.getenv(
        "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    dashscope_rerank_base_url: str = os.getenv(
        "DASHSCOPE_RERANK_BASE_URL", "https://dashscope.aliyuncs.com/compatible-api/v1"
    )

    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
    embedding_dim: int = _get_int("EMBEDDING_DIM", 1024)
    rerank_model: str = os.getenv("RERANK_MODEL", "qwen3-rerank")
    llm_model: str = os.getenv("LLM_MODEL", "qwen-plus")
    agentscope_max_iters: int = _get_int("AGENTSCOPE_MAX_ITERS", 4)
    intent_model_backend: str = os.getenv("INTENT_MODEL_BACKEND", "prompt_api")
    intent_confidence_threshold: float = _get_float(
        "INTENT_CONFIDENCE_THRESHOLD", 0.65
    )
    intent_fallback_to_rules: bool = _get_bool("INTENT_FALLBACK_TO_RULES", True)
    intent_api_model: str = os.getenv("INTENT_API_MODEL", "qwen-turbo")
    intent_api_base_url: str = os.getenv(
        "INTENT_API_BASE_URL",
        os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
    )
    intent_api_key: str = os.getenv(
        "INTENT_API_KEY", os.getenv("DASHSCOPE_API_KEY", "")
    )
    intent_local_model: str = os.getenv(
        "INTENT_LOCAL_MODEL", "Qwen2.5-0.5B-Instruct"
    )
    intent_local_base_url: str = os.getenv(
        "INTENT_LOCAL_BASE_URL", "http://127.0.0.1:8002/v1"
    )
    intent_local_api_key: str = os.getenv("INTENT_LOCAL_API_KEY", "local")
    intent_bert_model_path: str = _resolve_path(
        os.getenv("INTENT_BERT_MODEL_PATH", "models/intent_bert")
    )
    intent_bert_max_length: int = _get_int("INTENT_BERT_MAX_LENGTH", 384)
    max_image_inputs: int = _get_int("MAX_IMAGE_INPUTS", 10)
    ocr_languages: str = os.getenv("OCR_LANGUAGES", "ch_sim,en")
    ocr_model_storage_dir: str = _resolve_path(
        os.getenv("OCR_MODEL_STORAGE_DIR", "storage/easyocr_models")
    )
    image_analysis_tools: str = os.getenv("IMAGE_ANALYSIS_TOOLS", "ocr,api_vlm")
    api_vlm_model: str = os.getenv("API_VLM_MODEL", "qwen-vl-plus")
    api_vlm_base_url: str = os.getenv(
        "API_VLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    local_vlm_enabled: bool = _get_bool("LOCAL_VLM_ENABLED", False)
    local_vlm_base_url: str = os.getenv("LOCAL_VLM_BASE_URL", "http://127.0.0.1:8001/v1")
    local_vlm_model: str = os.getenv("LOCAL_VLM_MODEL", "")

    qdrant_url: str = os.getenv("QDRANT_URL", "")
    qdrant_path: str = DEFAULT_QDRANT_PATH
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "manual_chunks")

    vector_top_k: int = _get_int("VECTOR_TOP_K", 50)
    bm25_top_k: int = _get_int("BM25_TOP_K", 50)
    rrf_k: int = _get_int("RRF_K", 60)
    rerank_top_k: int = _get_int("RERANK_TOP_K", 8)
    final_context_top_k: int = _get_int("FINAL_CONTEXT_TOP_K", 6)

    storage_dir: str = DEFAULT_STORAGE_DIR
    manuals_dir: str = _resolve_path(os.getenv("MANUALS_DIR", "data/manuals"))

    enable_customer_qa_retrieval: bool = _get_bool(
        "ENABLE_CUSTOMER_QA_RETRIEVAL", False
    )
    customer_qa_path: str = _resolve_path(
        os.getenv("CUSTOMER_QA_PATH", "combined_qa_generic_optimized.json"),
        fallback_base=WORKSPACE_ROOT,
    )
    customer_qa_min_vector_score_gap: float = _get_float(
        "CUSTOMER_QA_MIN_VECTOR_SCORE_GAP", 1.1
    )
    customer_qa_qdrant_path: str = DEFAULT_CUSTOMER_QA_QDRANT_PATH
    customer_qa_query_collection: str = os.getenv(
        "CUSTOMER_QA_QUERY_COLLECTION", "customer_qa_query"
    )
    customer_qa_answer_collection: str = os.getenv(
        "CUSTOMER_QA_ANSWER_COLLECTION", "customer_qa_answer"
    )
    customer_qa_vector_top_k: int = _get_int("CUSTOMER_QA_VECTOR_TOP_K", 3)
    customer_qa_min_vector_score: float = _get_float(
        "CUSTOMER_QA_MIN_VECTOR_SCORE", 0.35
    )


def get_settings() -> Settings:
    settings = Settings()
    if not settings.dashscope_api_key:
        raise RuntimeError("请先设置环境变量 DASHSCOPE_API_KEY，或复制 .env.example 为 .env 后填写。")
    return settings
