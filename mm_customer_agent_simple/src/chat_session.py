from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Dict, List
from uuid import uuid4

from .agent import CustomerAgent


MAX_REWRITE_HISTORY_TURNS = 6


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ChatSessionManager:
    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._lock = RLock()

    def create_session(self) -> Dict[str, Any]:
        session_id = uuid4().hex
        session = {
            "session_id": session_id,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "messages": [],
        }
        with self._lock:
            self._sessions[session_id] = session
        return deepcopy(session)

    def get_session(self, session_id: str) -> Dict[str, Any] | None:
        with self._lock:
            session = self._sessions.get(session_id)
            return deepcopy(session) if session else None

    def ensure_session(self, session_id: str | None = None) -> Dict[str, Any]:
        if not session_id:
            return self.create_session()
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                return deepcopy(session)
        return self.create_session()

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        message = {
            "role": role,
            "content": content,
            "created_at": _utc_now(),
        }
        if metadata:
            message["metadata"] = metadata
        with self._lock:
            session = self._sessions.setdefault(
                session_id,
                {
                    "session_id": session_id,
                    "created_at": _utc_now(),
                    "updated_at": _utc_now(),
                    "messages": [],
                },
            )
            session["messages"].append(message)
            session["updated_at"] = _utc_now()
            return deepcopy(session)

    def list_messages(self, session_id: str) -> List[Dict[str, Any]]:
        session = self.get_session(session_id)
        return list(session.get("messages", [])) if session else []

    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)


class ChatAgentService:
    def __init__(self, agent: CustomerAgent, manager: ChatSessionManager):
        self.agent = agent
        self.manager = manager
        self._answer_lock = RLock()

    @staticmethod
    def _recent_rewrite_history(
        history: List[Dict[str, Any]],
        max_user_turns: int = MAX_REWRITE_HISTORY_TURNS,
    ) -> List[Dict[str, Any]]:
        if max_user_turns <= 0:
            return []

        start_index = 0
        user_turns = 0
        for index in range(len(history) - 1, -1, -1):
            role = str(history[index].get("role", "")).strip().lower()
            if role != "user":
                continue
            user_turns += 1
            if user_turns >= max_user_turns:
                start_index = index
                break
        return history[start_index:]

    def answer_chat_turn(
        self,
        session_id: str | None,
        message: str,
        debug: bool = False,
        flow_log: bool = True,
        image_paths: List[str] | None = None,
        image_base64: List[str] | None = None,
        turn_mode: str = "single",
    ) -> Dict[str, Any]:
        message = message.strip()
        if not message:
            raise ValueError("message must not be empty")

        session = self.manager.ensure_session(session_id)
        actual_session_id = session["session_id"]
        history = session.get("messages", [])
        rewrite_source_history = self._recent_rewrite_history(history)
        rewrite_history = []
        for item in rewrite_source_history:
            metadata = item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {}
            content = (
                metadata.get("final_query")
                or metadata.get("rewritten_query")
                or item.get("content", "")
            )
            rewrite_history.append({"role": item.get("role", ""), "content": content})

        rewritten_query = message
        with self._answer_lock:
            result = self.agent.answer(
                message,
                return_debug=debug,
                flow_log=flow_log,
                image_paths=image_paths,
                image_base64=image_base64,
                rewrite_history=rewrite_history,
                turn_mode=turn_mode,
            )

        answer = result.get("answer", "")
        self.manager.append_message(
            actual_session_id,
            "user",
            message,
            metadata={
                "rewritten_query": rewritten_query,
                "image_rewritten_query": result.get("image_rewritten_query", ""),
                "history_rewritten_query": result.get("history_rewritten_query", ""),
                "final_query": result.get("final_query", rewritten_query),
                "image_paths": image_paths or [],
                "has_image_base64": bool(image_base64),
            },
        )
        self.manager.append_message(
            actual_session_id,
            "assistant",
            answer,
            metadata={
                "image_ids": result.get("image_ids", []),
                "selected_sub_agent": result.get("selected_sub_agent", ""),
                "question_type": result.get("question_type", ""),
            },
        )
        updated_session = self.manager.get_session(actual_session_id) or {}

        return {
            "framework": result.get("framework", "agentscope"),
            "session_id": actual_session_id,
            "message": message,
            "rewritten_query": rewritten_query,
            "answer": answer,
            "answers": result.get("answers", []),
            "image_ids": result.get("image_ids", []),
            "selected_sub_agent": result.get("selected_sub_agent", ""),
            "question_type": result.get("question_type", ""),
            "route": result.get("route", ""),
            "image_rewritten_query": result.get("image_rewritten_query", ""),
            "history_rewritten_query": result.get("history_rewritten_query", ""),
            "final_query": result.get("final_query", ""),
            "actionability": result.get("actionability", {}),
            "main_plan": result.get("main_plan", {}),
            "tasks": result.get("tasks", []),
            "turn_results": result.get("turn_results", []),
            "task_results": result.get("task_results", []),
            "order_query": result.get("order_query", {}),
            "order_queries": result.get("order_queries", []),
            "agent_trace": result.get("agent_trace", []),
            "raw_result": result if debug else {},
            "history": updated_session.get("messages", []),
        }
