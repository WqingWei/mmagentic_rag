from __future__ import annotations

from typing import Any, Dict, List

from .agentscope_application import AgentScopeMultiAgentApplication
from .config import Settings


class CustomerAgent:
    """Public AgentScope multi-agent entrypoint used by every interface."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.runtime = AgentScopeMultiAgentApplication(settings)

    def __del__(self):
        try:
            self.runtime.close()
        except Exception:
            pass

    def answer(
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
        return self.runtime.answer_sync(
            question=question,
            return_debug=return_debug,
            question_id=question_id,
            flow_log=flow_log,
            image_paths=image_paths,
            image_base64=image_base64,
            rewrite_history=rewrite_history,
            turn_mode=turn_mode,
        )
