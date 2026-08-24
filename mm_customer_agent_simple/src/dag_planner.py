from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from .intent_model import IntentPrediction, IntentType


ALLOWED_NODE_TYPES = {
    "image_understanding",
    "knowledge",
    "customer_service",
    "clarification",
}


@dataclass(frozen=True)
class DAGNode:
    node_id: str
    node_type: str
    query: str
    depends_on: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExecutionPlan:
    intent: IntentType
    nodes: List[DAGNode]
    reason: str
    source: str
    summary_required: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent.value,
            "nodes": [node.to_dict() for node in self.nodes],
            "reason": self.reason,
            "source": self.source,
            "summary_required": self.summary_required,
        }

    def topological_nodes(self) -> List[DAGNode]:
        validate_plan(self)
        by_id = {node.node_id: node for node in self.nodes}
        indegree = {node.node_id: len(node.depends_on) for node in self.nodes}
        children: Dict[str, List[str]] = {node.node_id: [] for node in self.nodes}
        for node in self.nodes:
            for dependency in node.depends_on:
                children[dependency].append(node.node_id)
        ready = [node.node_id for node in self.nodes if indegree[node.node_id] == 0]
        ordered: List[DAGNode] = []
        while ready:
            current = ready.pop(0)
            ordered.append(by_id[current])
            for child in children[current]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        if len(ordered) != len(self.nodes):
            raise ValueError("DAG contains a cycle")
        return ordered


def validate_plan(plan: ExecutionPlan) -> None:
    if not plan.nodes:
        raise ValueError("DAG plan must contain at least one node")
    node_ids = [node.node_id for node in plan.nodes]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("DAG node IDs must be unique")
    known = set(node_ids)
    for node in plan.nodes:
        if node.node_type not in ALLOWED_NODE_TYPES:
            raise ValueError(f"unsupported DAG node type: {node.node_type}")
        if node.node_id in node.depends_on:
            raise ValueError(f"node cannot depend on itself: {node.node_id}")
        missing = set(node.depends_on) - known
        if missing:
            raise ValueError(f"unknown DAG dependencies: {sorted(missing)}")


def build_simple_plan(
    prediction: IntentPrediction,
    query: str,
    has_images: bool = False,
) -> ExecutionPlan:
    if prediction.intent == IntentType.MIXED:
        raise ValueError("mixed intent must be planned by Planner Agent")
    nodes: List[DAGNode] = []
    dependencies: List[str] = []
    if has_images:
        nodes.append(DAGNode("image_1", "image_understanding", query))
        dependencies = ["image_1"]
    node_type = {
        IntentType.KNOWLEDGE: "knowledge",
        IntentType.CUSTOMER_SERVICE: "customer_service",
        IntentType.CLARIFICATION: "clarification",
    }[prediction.intent]
    nodes.append(DAGNode("task_1", node_type, query, dependencies))
    plan = ExecutionPlan(
        intent=prediction.intent,
        nodes=nodes,
        reason=prediction.reason,
        source="intent_direct",
    )
    validate_plan(plan)
    return plan


def plan_from_json(value: str | Dict[str, Any], fallback_intent: IntentType) -> ExecutionPlan:
    data = json.loads(value) if isinstance(value, str) else value
    if not isinstance(data, dict):
        raise ValueError("planner output must be a JSON object")
    try:
        intent = IntentType(str(data.get("intent", fallback_intent.value)))
    except ValueError:
        intent = fallback_intent
    nodes = []
    for item in data.get("nodes", []) or []:
        nodes.append(
            DAGNode(
                node_id=str(item.get("node_id", "")).strip(),
                node_type=str(item.get("node_type", "")).strip(),
                query=str(item.get("query", "")).strip(),
                depends_on=[str(dep) for dep in item.get("depends_on", []) or []],
            )
        )
    plan = ExecutionPlan(
        intent=intent,
        nodes=nodes,
        reason=str(data.get("reason", "")).strip(),
        source="planner_agent",
        summary_required=bool(data.get("summary_required", True)),
    )
    validate_plan(plan)
    plan.topological_nodes()
    return plan
