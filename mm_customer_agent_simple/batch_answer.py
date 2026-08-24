#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

from tqdm import tqdm

from src.agent import CustomerAgent
from src.config import get_settings


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def _row_id(row: dict) -> int:
    try:
        return int(row["id"])
    except (KeyError, TypeError, ValueError):
        return -1


def _last_output_id(path: Path) -> int:
    if not path.exists():
        return -1
    last_id = -1
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_id = _row_id(row)
            if row_id > last_id:
                last_id = row_id
    return last_id


def _write_context_debug(f, row: dict, result: dict) -> None:
    f.write("=" * 100 + "\n")
    f.write(f"id: {row.get('id', '')}\n")
    f.write(f"route: {result.get('route', result.get('question_type', ''))}\n")
    f.write(f"question:\n{row.get('question', '')}\n\n")
    route_plan = result.get("route_plan") or {}
    if route_plan:
        f.write("route_plan:\n")
        f.write(json.dumps(route_plan, ensure_ascii=False, indent=2))
        f.write("\n\n")
        f.write(f"selected_sub_agent: {result.get('selected_sub_agent', '')}\n")
        f.write(f"sub_agent_result_type: {result.get('sub_agent_result_type', '')}\n\n")
    rewritten_turns = result.get("rewritten_turns")
    if rewritten_turns:
        f.write("rewritten_turns:\n")
        for idx, turn in enumerate(rewritten_turns, start=1):
            f.write(f"{idx}. {turn}\n")
        f.write("\n")
    customer_analyses = result.get("customer_analyses") or []
    if not customer_analyses and result.get("customer_analysis"):
        customer_analyses = [result["customer_analysis"]]
    if customer_analyses:
        f.write("customer_analyses:\n")
        for idx, analysis in enumerate(customer_analyses, start=1):
            f.write(f"{idx}. {json.dumps(analysis, ensure_ascii=False)}\n")
        f.write("\n")
    customer_qa_retrieval = result.get("customer_qa_retrieval") or []
    if customer_qa_retrieval:
        f.write("customer_qa_retrieval:\n")
        f.write(json.dumps(customer_qa_retrieval, ensure_ascii=False, indent=2))
        f.write("\n\n")
    f.write(f"answer:\n{result.get('answer', '')}\n")
    f.write(f"image_ids: {json.dumps(result.get('image_ids', []), ensure_ascii=False)}\n\n")

    retrieval_plans = result.get("retrieval_plans") or []
    if retrieval_plans:
        f.write("retrieval_plans:\n")
        for idx, plan in enumerate(retrieval_plans, start=1):
            f.write(f"{idx}. {json.dumps(plan, ensure_ascii=False)}\n")
        f.write("\n")

    followup_queries = result.get("followup_queries") or []
    if followup_queries:
        f.write("followup_queries:\n")
        for idx, query in enumerate(followup_queries, start=1):
            f.write(f"{idx}. {query}\n")
        f.write("\n")

    followup_searches = result.get("followup_searches") or []
    if followup_searches:
        f.write("followup_searches:\n")
        for search in followup_searches:
            f.write(
                f"round {search.get('round', '')} query: {search.get('query', '')}\n"
            )
            _write_chunk_list(f, "search_hits", search.get("chunks") or [])
        f.write("\n")

    initial_chunks = result.get("initial_chunks") or []
    if initial_chunks:
        _write_chunk_list(f, "initial_context", initial_chunks)

    followup_chunks = result.get("followup_chunks") or []
    if followup_chunks:
        _write_chunk_list(f, "followup_context", followup_chunks)

    candidate_chunks = result.get("candidate_chunks") or []
    if candidate_chunks:
        _write_chunk_list(f, "candidate_context_for_selection", candidate_chunks)

    evidence_selection = result.get("evidence_selection") or {}
    if evidence_selection:
        f.write("evidence_selection:\n")
        f.write(json.dumps(evidence_selection, ensure_ascii=False, indent=2))
        f.write("\n\n")

    chunks = result.get("chunks") or []
    if not chunks:
        f.write("model_context: direct LLM answer, no retrieval context.\n\n")
        return

    _write_chunk_list(f, "model_context", chunks)


def _write_chunk_list(f, title: str, chunks: list[dict]) -> None:
    f.write(f"{title}:\n")
    for idx, chunk in enumerate(chunks, start=1):
        f.write("-" * 80 + "\n")
        f.write(f"context_index: {idx}\n")
        f.write(f"manual_id: {chunk.get('manual_id', '')}\n")
        f.write(f"section_path: {chunk.get('section_path', '')}\n")
        f.write(f"score: {chunk.get('score', '')}\n")
        f.write(f"score_source: {chunk.get('score_source', '')}\n")
        f.write(f"rerank_score: {chunk.get('rerank_score', '')}\n")
        f.write(f"image_ids: {', '.join(chunk.get('image_ids') or []) or '无'}\n")
        f.write("text:\n")
        f.write(chunk.get("text", ""))
        f.write("\n")
    f.write("\n")


def _chunk_brief(chunk: dict) -> dict:
    text = (chunk.get("text") or "").replace("\n", " ").strip()
    if len(text) > 180:
        text = text[:180] + "..."
    return {
        "chunk_id": chunk.get("chunk_id", ""),
        "manual_id": chunk.get("manual_id", ""),
        "section_path": chunk.get("section_path", ""),
        "score": chunk.get("score", 0.0),
        "score_source": chunk.get("score_source", ""),
        "rerank_score": chunk.get("rerank_score"),
        "image_ids": chunk.get("image_ids") or [],
        "text_preview": text,
    }


def _build_trace_record(row: dict, result: dict) -> dict:
    return {
        "id": row.get("id", ""),
        "question": row.get("question", ""),
        "intent_prediction": result.get("intent_prediction") or {},
        "execution_plan": result.get("execution_plan") or {},
        "route": result.get("route", result.get("question_type", "")),
        "question_type": result.get("question_type", ""),
        "route_plan": result.get("route_plan") or {},
        "main_plan": result.get("main_plan") or {},
        "tasks": result.get("tasks") or [],
        "task_results": result.get("task_results") or [],
        "turn_results": result.get("turn_results") or [],
        "agent_trace": result.get("agent_trace") or [],
        "selected_sub_agent": result.get("selected_sub_agent", ""),
        "sub_agent_result_type": result.get("sub_agent_result_type", ""),
        "rewritten_turns": result.get("rewritten_turns") or [],
        "customer_analyses": result.get("customer_analyses") or [],
        "customer_analysis": result.get("customer_analysis") or {},
        "customer_qa_retrieval": result.get("customer_qa_retrieval") or [],
        "answer_sources": result.get("answer_sources") or [],
        "initial_query": result.get("initial_query", ""),
        "initial_chunks": [
            _chunk_brief(chunk) for chunk in result.get("initial_chunks") or []
        ],
        "retrieval_plans": result.get("retrieval_plans") or [],
        "followup_queries": result.get("followup_queries") or [],
        "followup_searches": [
            {
                "round": search.get("round", ""),
                "query": search.get("query", ""),
                "chunks": [
                    _chunk_brief(chunk) for chunk in search.get("chunks") or []
                ],
            }
            for search in result.get("followup_searches") or []
        ],
        "candidate_chunks": [
            _chunk_brief(chunk) for chunk in result.get("candidate_chunks") or []
        ],
        "evidence_selection": result.get("evidence_selection") or {},
        "final_chunks": [
            _chunk_brief(chunk) for chunk in result.get("chunks") or []
        ],
        "answer": result.get("answer", ""),
        "image_ids": result.get("image_ids", []),
    }


def _print_retrieval_trace(row: dict, result: dict, max_chunks: int = 3) -> None:
    route_plan = result.get("route_plan") or {}
    if route_plan:
        tqdm.write(
            "[主路由] "
            f"intent={route_plan.get('intent', '')} "
            f"source={route_plan.get('source', '')} "
            f"confidence={route_plan.get('confidence', '')} "
            f"override={route_plan.get('rule_override', False)}"
        )
    if result.get("route") != "agentic_retrieval":
        tqdm.write("[检索] direct LLM，无手册检索")
        analyses = result.get("customer_analyses") or []
        if not analyses and result.get("customer_analysis"):
            analyses = [result["customer_analysis"]]
        for idx, analysis in enumerate(analyses, start=1):
            sub_questions = analysis.get("sub_questions") or []
            tqdm.write(
                "[客服分析] "
                f"第{idx}轮 main={analysis.get('main_intent', '')} "
                f"sub_questions={json.dumps(sub_questions, ensure_ascii=False)}"
            )
        return

    tqdm.write(f"[检索] 初始 query: {result.get('initial_query') or row.get('question', '')}")
    for idx, chunk in enumerate((result.get("initial_chunks") or [])[:max_chunks], start=1):
        brief = _chunk_brief(chunk)
        tqdm.write(
            f"  初始{idx}: {brief['manual_id']} | {brief['section_path']} | "
            f"rerank={brief['rerank_score']}"
        )

    plans = result.get("retrieval_plans") or []
    for idx, plan in enumerate(plans, start=1):
        tqdm.write(
            "[规划] "
            f"第{idx}轮 sufficient={plan.get('evidence_sufficient')} "
            f"queries={json.dumps(plan.get('search_queries', []), ensure_ascii=False)} "
            f"reason={plan.get('reason', '')}"
        )

    searches = result.get("followup_searches") or []
    if searches:
        for search in searches:
            tqdm.write(f"[补查] round={search.get('round')} query={search.get('query')}")
            for idx, chunk in enumerate((search.get("chunks") or [])[:max_chunks], start=1):
                brief = _chunk_brief(chunk)
                tqdm.write(
                    f"  补查{idx}: {brief['manual_id']} | {brief['section_path']} | "
                    f"rerank={brief['rerank_score']}"
                )
    else:
        tqdm.write("[补查] 未触发补充检索")

    selection = result.get("evidence_selection") or {}
    if selection:
        selected_ids = selection.get("selected_chunk_ids") or []
        tqdm.write(
            "[选证] "
            f"fallback={selection.get('fallback', False)} "
            f"selected={len(selected_ids)} "
            f"reason={selection.get('reason', '')}"
        )
        by_id = {
            chunk.get("chunk_id"): chunk
            for chunk in result.get("candidate_chunks") or []
        }
        for idx, chunk_id in enumerate(selected_ids[:max_chunks], start=1):
            brief = _chunk_brief(by_id.get(chunk_id, {}))
            tqdm.write(
                f"  选证{idx}: {brief['manual_id']} | {brief['section_path']} | "
                f"rerank={brief['rerank_score']}"
            )

    final_chunks = result.get("chunks") or []
    tqdm.write(f"[最终证据] {len(final_chunks)} 条")
    for idx, chunk in enumerate(final_chunks[:max_chunks], start=1):
        brief = _chunk_brief(chunk)
        tqdm.write(
            f"  最终{idx}: {brief['manual_id']} | {brief['section_path']} | "
            f"rerank={brief['rerank_score']}"
        )


def _clean_question_turn(text: str) -> str:
    value = text.strip()
    if value.endswith(","):
        value = value[:-1].strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        value = value[1:-1]
    return value.replace('""', '"').strip()


def _split_question_turns(question: str) -> list[str]:
    turns = []
    for line in question.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        turn = _clean_question_turn(line)
        if turn:
            turns.append(turn)
    if turns:
        return turns
    question = _clean_question_turn(question)
    return [question] if question else []


def _normalize_answer_text(answer: str) -> str:
    while "\\\\n" in answer:
        answer = answer.replace("\\\\n", "\\n")
    while "\\\\r" in answer:
        answer = answer.replace("\\\\r", "\\r")
    return answer


def _format_manual_ret(answer: str, image_ids: list) -> str:
    answer = _normalize_answer_text(answer)
    return f"{json.dumps(answer, ensure_ascii=False)},{json.dumps(image_ids, ensure_ascii=False)}"


def _format_customer_ret(answers: list[str]) -> str:
    answers = [_normalize_answer_text(answer) for answer in answers]
    return ",".join(json.dumps(answer, ensure_ascii=False) for answer in answers)


def _format_mixed_ret(result: dict) -> str:
    turn_answers = result.get("answers") or [
        turn.get("answer", "")
        for turn in result.get("turn_results") or []
        if turn.get("answer")
    ]
    if not turn_answers:
        turn_answers = [result.get("answer", "")]
    turn_answers = [_normalize_answer_text(answer) for answer in turn_answers]
    answer_part = ",".join(
        json.dumps(answer, ensure_ascii=False) for answer in turn_answers
    )
    image_part = json.dumps(result.get("image_ids", []), ensure_ascii=False)
    return f"{answer_part},{image_part}"


def main():
    _configure_stdout()
    parser = argparse.ArgumentParser(description="批量回答赛题 CSV。输入 CSV 至少包含 id, question 两列。")
    parser.add_argument("--input", required=True, help="输入 questions.csv")
    parser.add_argument("--output", required=True, help="输出 answers.csv")
    parser.add_argument("--start-id", type=int, default=0, help="只处理 id 大于等于该值的题；0 表示不跳过")
    parser.add_argument("--end-id", type=int, default=0, help="只处理 id 小于等于该值的题；0 表示不限制")
    parser.add_argument("--context-output", default="", help="可选：导出每题给模型的检索上下文，便于调试")
    parser.add_argument("--trace-output", default="", help="导出 agent 检索轨迹 JSONL；默认写到 output 同名 .trace.jsonl")
    parser.add_argument("--no-trace", action="store_true", help="不打印检索轨迹，也不生成 trace 文件")
    parser.add_argument("--resume", action="store_true", help="从已有 output 的最后 id 后继续，并追加写入")
    parser.add_argument("--ret-as-json", action="store_true", help="兼容旧参数；当前不会改变输出格式")
    parser.add_argument(
        "--no-agent-flow-log",
        action="store_true",
        help="不在终端打印 AgentScope 意图/规划/DAG/Agent 流程",
    )
    args = parser.parse_args()

    settings = get_settings()
    agent = CustomerAgent(settings)

    in_path = Path(args.input)
    out_path = Path(args.output)
    resume_last_id = _last_output_id(out_path) if args.resume else -1
    if args.resume and resume_last_id >= 0:
        args.start_id = max(args.start_id, resume_last_id + 1)
        print(f"续跑模式：已有输出最后 id={resume_last_id}，本次从 id={args.start_id} 开始。")

    rows = list(csv.DictReader(in_path.open("r", encoding="utf-8-sig")))
    if args.start_id > 0:
        rows = [row for row in rows if _row_id(row) >= args.start_id]
    if args.end_id > 0:
        rows = [row for row in rows if _row_id(row) <= args.end_id]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    context_path = Path(args.context_output) if args.context_output else None
    context_f = None
    if context_path:
        context_path.parent.mkdir(parents=True, exist_ok=True)
        context_mode = "a" if args.resume and context_path.exists() else "w"
        context_f = context_path.open(context_mode, encoding="utf-8", newline="\r\n")
    trace_path = None
    trace_f = None
    if not args.no_trace:
        trace_path = Path(args.trace_output) if args.trace_output else out_path.with_suffix(".trace.jsonl")
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_mode = "a" if args.resume and trace_path.exists() else "w"
        trace_f = trace_path.open(trace_mode, encoding="utf-8", newline="\n")

    output_mode = "a" if args.resume and out_path.exists() else "w"
    with out_path.open(output_mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "ret"], lineterminator="\r\n")
        if output_mode == "w":
            writer.writeheader()
        for row in tqdm(rows):
            q = row["question"]
            tqdm.write(f"\n[输入] id={row['id']}\n{q}")
            result = agent.answer(
                q,
                return_debug=bool(context_f) or bool(trace_f),
                question_id=row.get("id", ""),
                flow_log=not args.no_agent_flow_log,
            )
            if result.get("question_type") == "customer_service":
                ret = _format_customer_ret(result.get("answers") or [result["answer"]])
            elif result.get("question_type") == "mixed":
                ret = _format_mixed_ret(result)
            else:
                ret = _format_manual_ret(result["answer"], result["image_ids"])
            tqdm.write(f"[路线] {result.get('question_type', 'unknown')}")
            if trace_f and not result.get("agent_trace"):
                _print_retrieval_trace(row, result)
            tqdm.write(f"[输出] id={row['id']}\n{ret}\n")
            writer.writerow({"id": row["id"], "ret": ret})
            if context_f:
                _write_context_debug(context_f, row, result)
            if trace_f:
                trace_record = _build_trace_record(row, result)
                trace_f.write(json.dumps(trace_record, ensure_ascii=False) + "\n")

    if context_f:
        context_f.close()
    if trace_f:
        trace_f.close()

    print(f"完成: {out_path}")
    if context_path:
        print(f"上下文调试文件: {context_path}")
    if trace_path:
        print(f"检索轨迹文件: {trace_path}")


if __name__ == "__main__":
    main()
