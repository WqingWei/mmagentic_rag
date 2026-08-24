from typing import Dict, List

from .config import Settings
from .customer_qa_data import (
    build_customer_qa_manifest,
    load_customer_qa_entries,
    manifest_matches,
    read_customer_qa_manifest,
    resolve_customer_qa_path,
)
from .customer_qa_vector_store import CustomerQAVectorStore
from .embedding import EmbeddingClient


class CustomerQARetriever:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.top_k = max(1, settings.customer_qa_vector_top_k)
        self.min_score = settings.customer_qa_min_vector_score
        self.min_score_gap = settings.customer_qa_min_vector_score_gap
        self.path = resolve_customer_qa_path(settings.customer_qa_path)
        self.entries, raw_count, deduped_count = load_customer_qa_entries(self.path)
        if not self.entries:
            raise ValueError(f"customer QA file has no valid entries: {self.path}")
        self.entries_by_index = {entry.index: entry for entry in self.entries}
        self.embedder = EmbeddingClient(settings)
        self.store: CustomerQAVectorStore | None = None
        self.ready = self._load_vector_index(raw_count, deduped_count)
        print(
            "[customer_qa] "
            f"enabled=true mode=embedding path={self.path} entries={raw_count} "
            f"deduped={deduped_count} indexed={len(self.entries)} "
            f"ready={str(self.ready).lower()}"
        )

    def answer(
        self,
        query: str,
        turn_index: int = 1,
        turn_total: int = 1,
        original_query: str = "",
    ) -> Dict:
        turn_label = self._turn_label(turn_index, turn_total)
        original_query = original_query or query
        print(f"[customer_qa][turn={turn_label}] query={query!r}")

        if not self.ready or self.store is None:
            reason = "customer_qa_vector_index_unavailable"
            print(
                f"[customer_qa][turn={turn_label}] "
                f"final=direct_llm reason={reason}"
            )
            empty_q2q = self._empty_strategy("query2query", query, reason)
            empty_q2a = self._empty_strategy("query2answer", query, reason)
            return self._fallback_result(
                query=query,
                turn_index=turn_index,
                turn_total=turn_total,
                original_query=original_query,
                reason=reason,
                query2query=empty_q2q,
                query2answer=empty_q2a,
            )

        query_vector = self.embedder.embed_query(query)
        query2query = self._search_strategy(
            "query2query",
            query,
            query_vector,
            turn_label=turn_label,
            next_on_miss="query2answer",
        )
        if query2query["hit"]:
            query2answer = {
                "strategy": "query2answer",
                "query": query,
                "top_k": self.top_k,
                "turn_index": turn_index,
                "turn_total": turn_total,
                "hit": False,
                "skipped": True,
                "reason": "query2query_hit",
                "candidates": [],
            }
            print(f"[query2answer][turn={turn_label}] skipped reason=query2query_hit")
            print(
                f"[customer_qa][turn={turn_label}] "
                "final=faq_answer source=query2query"
            )
            return {
                "hit": True,
                "strategy": "query2query",
                "answer": query2query["answer"],
                "turn_index": turn_index,
                "turn_total": turn_total,
                "original_query": original_query,
                "final_source": "query2query",
                "final_reason": "query2query_hit",
                "query2query": query2query,
                "query2answer": query2answer,
            }

        query2answer = self._search_strategy(
            "query2answer",
            query,
            query_vector,
            turn_label=turn_label,
            next_on_miss="direct_llm",
        )
        if query2answer["hit"]:
            print(
                f"[customer_qa][turn={turn_label}] "
                "final=faq_answer source=query2answer"
            )
            return {
                "hit": True,
                "strategy": "query2answer",
                "answer": query2answer["answer"],
                "turn_index": turn_index,
                "turn_total": turn_total,
                "original_query": original_query,
                "final_source": "query2answer",
                "final_reason": "query2answer_hit",
                "query2query": query2query,
                "query2answer": query2answer,
            }

        print(
            f"[customer_qa][turn={turn_label}] "
            "final=direct_llm reason=no_confident_faq_hit"
        )
        return self._fallback_result(
            query=query,
            turn_index=turn_index,
            turn_total=turn_total,
            original_query=original_query,
            reason="no_confident_faq_hit",
            query2query=query2query,
            query2answer=query2answer,
        )

    def _load_vector_index(self, raw_count: int, deduped_count: int) -> bool:
        expected = build_customer_qa_manifest(
            self.settings,
            self.path,
            raw_count=raw_count,
            deduped_count=deduped_count,
            indexed_count=len(self.entries),
        )
        actual = read_customer_qa_manifest(self.settings)
        if not manifest_matches(expected, actual):
            print(
                "[customer_qa] vector index unavailable or stale. "
                "Run: python ingest_customer_qa.py"
            )
            return False
        try:
            self.store = CustomerQAVectorStore(self.settings)
            return True
        except Exception as exc:
            print(f"[customer_qa] failed to open vector index: {exc}")
            self.store = None
            return False

    def close(self) -> None:
        if self.store is not None:
            self.store.close()

    def _search_strategy(
        self,
        strategy: str,
        query: str,
        query_vector: List[float],
        turn_label: str,
        next_on_miss: str,
    ) -> Dict:
        assert self.store is not None
        raw_hits = (
            self.store.search_query(query_vector, self.top_k)
            if strategy == "query2query"
            else self.store.search_answer(query_vector, self.top_k)
        )
        candidates = []
        for payload, score in raw_hits:
            source_index = payload.get("source_index")
            try:
                source_index = int(source_index)
            except (TypeError, ValueError):
                continue
            if source_index not in self.entries_by_index:
                continue
            entry = self.entries_by_index[source_index]
            candidates.append(
                self._build_candidate(strategy, entry, len(candidates) + 1, score)
            )

        top_score = candidates[0]["vector_score"] if candidates else 0.0
        second_score = candidates[1]["vector_score"] if len(candidates) > 1 else 0.0
        hit, reason = self._is_confident_hit(top_score, second_score)
        selected_answer = self._candidate_answer(strategy, candidates[0]) if hit else ""
        status = "hit" if hit else "miss"
        miss_suffix = ""
        if not hit:
            miss_suffix = f" next={next_on_miss}"
            if next_on_miss == "direct_llm":
                miss_suffix += " fallback=direct_llm"
        print(
            f"[{strategy}][turn={turn_label}] "
            f"status={status} reason={reason} "
            f"top_vector_score={top_score:.4f} "
            f"second_vector_score={second_score:.4f}{miss_suffix}"
        )
        for candidate in candidates:
            self._print_candidate(strategy, candidate, turn_label)
        if hit:
            self._print_selected(strategy, candidates[0], turn_label)

        return {
            "strategy": strategy,
            "query": query,
            "top_k": self.top_k,
            "hit": hit,
            "skipped": False,
            "score": top_score,
            "vector_score": top_score,
            "top_score": top_score,
            "top_vector_score": top_score,
            "second_score": second_score,
            "second_vector_score": second_score,
            "reason": reason,
            "answer": selected_answer,
            "candidates": candidates,
        }

    @staticmethod
    def _build_candidate(strategy: str, entry, rank: int, score: float) -> Dict:
        if strategy == "query2query":
            return {
                "rank": rank,
                "score": score,
                "vector_score": score,
                "matched_question": entry.question,
                "mapped_answer": entry.answer,
                "source_index": entry.index,
            }
        return {
            "rank": rank,
            "score": score,
            "vector_score": score,
            "matched_answer": entry.answer,
            "source_question": entry.question,
            "source_index": entry.index,
        }

    @staticmethod
    def _candidate_answer(strategy: str, candidate: Dict) -> str:
        if strategy == "query2query":
            return candidate.get("mapped_answer", "")
        return candidate.get("matched_answer", "")

    def _print_candidate(self, strategy: str, candidate: Dict, turn_label: str) -> None:
        if strategy == "query2query":
            print(
                f"[query2query][turn={turn_label}] "
                f"candidate#{candidate['rank']} "
                f"vector_score={candidate['vector_score']:.4f} "
                f"matched_question={candidate['matched_question']!r} "
                f"mapped_answer={self._preview(candidate['mapped_answer'])!r}"
            )
            return
        print(
            f"[query2answer][turn={turn_label}] "
            f"candidate#{candidate['rank']} "
            f"vector_score={candidate['vector_score']:.4f} "
            f"matched_answer={self._preview(candidate['matched_answer'])!r} "
            f"source_question={candidate['source_question']!r}"
        )

    def _print_selected(self, strategy: str, candidate: Dict, turn_label: str) -> None:
        if strategy == "query2query":
            print(
                f"[query2query][turn={turn_label}] "
                f"selected_question={candidate['matched_question']!r} "
                f"selected_answer={self._preview(candidate['mapped_answer'])!r}"
            )
            return
        print(
            f"[query2answer][turn={turn_label}] "
            f"selected_answer={self._preview(candidate['matched_answer'])!r} "
            f"source_question={candidate['source_question']!r}"
        )

    @staticmethod
    def _preview(text: str, max_chars: int = 120) -> str:
        value = " ".join(text.split())
        if len(value) <= max_chars:
            return value
        return value[:max_chars] + "..."

    @staticmethod
    def _turn_label(turn_index: int, turn_total: int) -> str:
        return f"{turn_index}/{max(1, turn_total)}"

    def _is_confident_hit(
        self, top_score: float, second_score: float
    ) -> tuple[bool, str]:
        if top_score <= 0:
            return False, "no_candidates"
        if top_score < self.min_score:
            return False, "vector_score_below_threshold"
        if second_score > 0 and top_score / second_score < self.min_score_gap:
            return False, "score_gap_below_threshold"
        return True, "threshold_pass"

    @staticmethod
    def _empty_strategy(strategy: str, query: str, reason: str) -> Dict:
        return {
            "strategy": strategy,
            "query": query,
            "top_k": 0,
            "hit": False,
            "skipped": True,
            "score": 0.0,
            "vector_score": 0.0,
            "top_score": 0.0,
            "top_vector_score": 0.0,
            "second_score": 0.0,
            "second_vector_score": 0.0,
            "reason": reason,
            "answer": "",
            "candidates": [],
        }

    @staticmethod
    def _fallback_result(
        query: str,
        turn_index: int,
        turn_total: int,
        original_query: str,
        reason: str,
        query2query: Dict,
        query2answer: Dict,
    ) -> Dict:
        return {
            "hit": False,
            "strategy": "",
            "answer": "",
            "fallback": "direct_llm",
            "turn_index": turn_index,
            "turn_total": turn_total,
            "original_query": original_query,
            "final_source": "direct_llm",
            "final_reason": reason,
            "query2query": query2query,
            "query2answer": query2answer,
        }
