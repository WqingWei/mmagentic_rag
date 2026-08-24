#!/usr/bin/env python3
import argparse
import json
import sys

from src.agent import CustomerAgent
from src.config import get_settings


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def main():
    _configure_stdout()
    parser = argparse.ArgumentParser(description="Single-question customer agent QA.")
    parser.add_argument("--question", required=True, help="User question")
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="Optional image path. Can be provided multiple times.",
    )
    parser.add_argument("--debug", action="store_true", help="Print debug evidence")
    parser.add_argument(
        "--turn-mode",
        choices=["auto_newline", "single"],
        default="auto_newline",
        help="How to split turns in the question.",
    )
    parser.add_argument(
        "--no-agent-flow-log",
        action="store_true",
        help="Do not print AgentScope intent/planner/DAG/agent flow logs",
    )
    args = parser.parse_args()

    settings = get_settings()
    agent = CustomerAgent(settings)
    try:
        result = agent.answer(
            args.question,
            return_debug=args.debug,
            flow_log=not args.no_agent_flow_log,
            image_paths=args.image,
            turn_mode=args.turn_mode,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2))
        sys.exit(1)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
