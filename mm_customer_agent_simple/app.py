from __future__ import annotations

import base64
import json
import mimetypes
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import quote

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

try:
    from .src.agent import CustomerAgent
    from .src.chat_session import ChatAgentService, ChatSessionManager
    from .src.config import get_settings
except ImportError:
    from src.agent import CustomerAgent
    from src.chat_session import ChatAgentService, ChatSessionManager
    from src.config import get_settings


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


_configure_stdout()


class AnswerRequest(BaseModel):
    question: str = Field(..., min_length=1)
    question_id: str = ""
    debug: bool = True
    flow_log: bool = False
    turn_mode: str = "auto_newline"
    image_paths: List[str] = Field(default_factory=list)
    image_base64: List[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str = Field(..., min_length=1)
    debug: bool = True
    flow_log: bool = False
    turn_mode: str = "single"
    image_paths: List[str] = Field(default_factory=list)
    image_base64: List[str] = Field(default_factory=list)


class SessionCreateRequest(BaseModel):
    pass


settings = get_settings()
agent = CustomerAgent(settings)
session_manager = ChatSessionManager()
chat_service = ChatAgentService(agent, session_manager)

app = FastAPI(title="MM Customer Agent", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PACKAGE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PACKAGE_ROOT.parent
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".gif")
CHATBOT_CSS = """
.mm-chatbot img {
    max-width: 220px !important;
    max-height: 180px !important;
    object-fit: contain;
    border-radius: 6px;
}
"""


def _json(data: Any) -> Any:
    return jsonable_encoder(data)


def _image_asset_dirs() -> List[Path]:
    raw_dirs = os.getenv("IMAGE_ASSET_DIRS", "")
    dirs: List[Path] = []
    if raw_dirs.strip():
        dirs.extend(Path(item.strip()) for item in raw_dirs.split(";") if item.strip())
    dirs.extend(
        [
            PACKAGE_ROOT / "data" / "插图",
            PACKAGE_ROOT / "data" / "images",
            PACKAGE_ROOT / "images",
            PACKAGE_ROOT / "assets" / "images",
            WORKSPACE_ROOT / "images",
            WORKSPACE_ROOT / "assets" / "images",
        ]
    )

    resolved: List[Path] = []
    for path in dirs:
        candidate = path if path.is_absolute() else WORKSPACE_ROOT / path
        if candidate.exists() and candidate.is_dir() and candidate not in resolved:
            resolved.append(candidate)
    return resolved


def _resolve_image_path(image_id: str) -> Path | None:
    safe_id = Path(image_id).name
    if not safe_id:
        return None
    for directory in _image_asset_dirs():
        for ext in IMAGE_EXTENSIONS:
            candidate = directory / f"{safe_id}{ext}"
            if candidate.exists() and candidate.is_file():
                return candidate
    return None


def _image_paths(image_ids: List[str]) -> List[str]:
    paths: List[str] = []
    for image_id in image_ids:
        path = _resolve_image_path(image_id)
        if path is not None:
            paths.append(str(path))
    return paths


def _image_urls(image_ids: List[str]) -> List[Dict[str, str]]:
    urls = []
    for image_id in image_ids:
        if _resolve_image_path(image_id) is not None:
            urls.append({"image_id": image_id, "url": _image_url(image_id)})
    return urls


def _image_url(image_id: str) -> str:
    return f"/api/images/{quote(image_id)}"


def _image_markdown(image_id: str) -> str:
    if _resolve_image_path(image_id) is None:
        return image_id
    return f"![{image_id}]({_image_url(image_id)})"


def _answer_with_inline_images(answer: str, image_ids: List[str]) -> str:
    if not image_ids:
        return answer

    remaining = list(image_ids)
    content = answer or ""
    while "<PIC>" in content and remaining:
        image_id = remaining.pop(0)
        content = content.replace("<PIC>", f"\n\n{_image_markdown(image_id)}\n\n", 1)

    if remaining:
        images = "\n\n".join(_image_markdown(image_id) for image_id in remaining)
        content = f"{content.rstrip()}\n\n{images}" if content.strip() else images
    return content.strip()


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "agent_framework": "AgentScope",
        "llm_model": settings.llm_model,
        "intent_model_backend": settings.intent_model_backend,
        "enable_customer_qa_retrieval": settings.enable_customer_qa_retrieval,
        "session_count": session_manager.session_count(),
    }


@app.get("/api/images/{image_id}")
def get_image(image_id: str) -> FileResponse:
    image_path = _resolve_image_path(image_id)
    if image_path is None:
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(image_path)


@app.post("/api/answer")
def answer_once(request: AnswerRequest) -> Any:
    try:
        result = agent.answer(
            request.question,
            return_debug=request.debug,
            question_id=request.question_id,
            flow_log=request.flow_log,
            image_paths=request.image_paths,
            image_base64=request.image_base64,
            turn_mode=request.turn_mode,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    result["image_assets"] = _image_urls(result.get("image_ids", []))
    return _json(result)


@app.post("/api/sessions")
def create_session(_: SessionCreateRequest | None = None) -> Any:
    return _json(session_manager.create_session())


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> Any:
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    return _json(session)


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str) -> Dict[str, Any]:
    deleted = session_manager.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="session not found")
    return {"ok": True, "session_id": session_id}


@app.post("/api/chat")
def chat(request: ChatRequest) -> Any:
    try:
        result = chat_service.answer_chat_turn(
            request.session_id,
            request.message,
            debug=request.debug,
            flow_log=request.flow_log,
            image_paths=request.image_paths,
            image_base64=request.image_base64,
            turn_mode=request.turn_mode,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    result["image_assets"] = _image_urls(result.get("image_ids", []))
    return _json(result)


def _dump_debug(value: Any) -> str:
    return json.dumps(_json(value), ensure_ascii=False, indent=2)


def _summarize_routes(result: Dict[str, Any]) -> str:
    tasks = result.get("tasks") or []
    if not tasks:
        return ""
    lines = []
    for task in tasks:
        lines.append(
            f"task={task.get('task_id')} turn={task.get('turn_index')} "
            f"agent={task.get('sub_agent')} query={task.get('query')}"
        )
    return "\n".join(lines)


def _uploaded_image_paths(files: Any) -> List[str]:
    if not files:
        return []
    items = files if isinstance(files, list) else [files]
    paths: List[str] = []
    for item in items:
        if isinstance(item, str):
            paths.append(item)
            continue
        name = getattr(item, "name", "")
        if name:
            paths.append(str(name))
            continue
        path = getattr(item, "path", "")
        if path:
            paths.append(str(path))
    return paths


def _uploaded_images_markdown(image_paths: List[str]) -> str:
    parts: List[str] = []
    for idx, raw_path in enumerate(image_paths, start=1):
        path = Path(raw_path)
        if not path.exists() or not path.is_file():
            continue
        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        try:
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        except OSError:
            continue
        parts.append(
            f'<img src="data:{mime_type};base64,{encoded}" '
            f'alt="uploaded image {idx}" style="max-width:220px;max-height:180px;'
            'object-fit:contain;border-radius:6px;margin-top:6px;" />'
        )
    return "\n".join(parts)


def _chat_user_content(user_message: str, image_paths: List[str]) -> str:
    text = user_message.strip()
    image_markdown = _uploaded_images_markdown(image_paths)
    if text and image_markdown:
        return f"{text}\n\n{image_markdown}"
    if image_markdown:
        return image_markdown
    return text


def _build_gradio_app():
    try:
        import gradio as gr
    except ImportError:
        return None

    with gr.Blocks(title="MM Customer Agent") as demo:
        gr.HTML(f"<style>{CHATBOT_CSS}</style>")
        gr.Markdown("# MM Customer Agent")
        session_state = gr.State("")
        chatbot = gr.Chatbot(
            label="Conversation",
            height=420,
            elem_classes=["mm-chatbot"],
        )
        message = gr.Textbox(
            label="Message",
            placeholder="输入当前轮问题，例如：多久能收到？",
            lines=2,
        )
        images = gr.File(
            label="Images",
            file_count="multiple",
            file_types=["image"],
            type="filepath",
        )
        send_btn = gr.Button("Send", variant="primary")

        def chat_submit(
            user_message: str,
            image_files: Any,
            chat_history: List[Dict[str, str]] | None,
            session_id: str,
        ):
            chat_history = chat_history or []
            image_paths = _uploaded_image_paths(image_files)
            if not user_message.strip() and not image_paths:
                return (
                    chat_history,
                    session_id,
                    "",
                    None,
                )
            try:
                result = chat_service.answer_chat_turn(
                    session_id or None,
                    user_message or "请根据我上传的图片进行回答。",
                    debug=False,
                    flow_log=bool(image_paths),
                    image_paths=image_paths,
                    turn_mode="single",
                )
            except (ValueError, FileNotFoundError, RuntimeError) as exc:
                raise gr.Error(str(exc)) from exc
            result["image_assets"] = _image_urls(result.get("image_ids", []))
            assistant_message = _answer_with_inline_images(
                result.get("answer", ""),
                result.get("image_ids", []),
            )
            updated_history = chat_history + [
                {
                    "role": "user",
                    "content": _chat_user_content(user_message, image_paths),
                },
                {"role": "assistant", "content": assistant_message},
            ]
            return (
                updated_history,
                result.get("session_id", ""),
                "",
                None,
            )

        send_btn.click(
            chat_submit,
            inputs=[message, images, chatbot, session_state],
            outputs=[
                chatbot,
                session_state,
                message,
                images,
            ],
        )
        message.submit(
            chat_submit,
            inputs=[message, images, chatbot, session_state],
            outputs=[
                chatbot,
                session_state,
                message,
                images,
            ],
        )

    return demo


gradio_app = _build_gradio_app()
if gradio_app is not None:
    import gradio as gr

    app = gr.mount_gradio_app(app, gradio_app, path="/")
else:

    @app.get("/")
    def gradio_missing() -> Dict[str, str]:
        return {
            "message": "FastAPI is running. Install gradio and restart to enable the web UI.",
            "install": "pip install -r requirements.txt",
            "docs": "/docs",
        }
