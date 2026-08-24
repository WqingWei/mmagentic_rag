from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List


ORDER_QUERY_TERMS = (
    "订单",
    "物流",
    "快递",
    "包裹",
    "发货",
    "配送",
    "签收",
    "order",
    "shipment",
    "shipping",
    "delivery",
    "tracking",
    "parcel",
)
ORDER_ID_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:ORD|ORDER|DD|MOCK)[-_]?[A-Za-z0-9-]{6,24}(?![A-Za-z0-9])",
    re.IGNORECASE,
)
NUMERIC_ORDER_ID_RE = re.compile(r"(?<!\d)\d{12,20}(?!\d)")


@dataclass(frozen=True)
class MockOrder:
    order_id: str
    status: str
    status_text: str
    product_name: str
    quantity: int
    amount: float
    created_at: str
    paid_at: str
    shipped_at: str | None
    delivered_at: str | None
    carrier: str | None
    tracking_number: str | None
    receiver: str
    receiver_phone: str
    receiver_address: str
    latest_event: str


DEFAULT_MOCK_ORDERS = (
    MockOrder(
        order_id="ORD202608230001",
        status="shipped",
        status_text="运输中",
        product_name="智能空气净化器 Pro",
        quantity=1,
        amount=1299.00,
        created_at="2026-08-20 09:15:00",
        paid_at="2026-08-20 09:16:12",
        shipped_at="2026-08-21 14:30:00",
        delivered_at=None,
        carrier="顺丰速运",
        tracking_number="SF MOCK 202608230001",
        receiver="王*",
        receiver_phone="138****1234",
        receiver_address="上海市浦东新区****路**号",
        latest_event="2026-08-23 08:40 快件已到达上海浦东集散中心",
    ),
    MockOrder(
        order_id="ORD202608230002",
        status="pending_shipment",
        status_text="待发货",
        product_name="人体工学椅 E2",
        quantity=1,
        amount=1899.00,
        created_at="2026-08-22 20:08:00",
        paid_at="2026-08-22 20:09:03",
        shipped_at=None,
        delivered_at=None,
        carrier=None,
        tracking_number=None,
        receiver="李*",
        receiver_phone="186****5678",
        receiver_address="杭州市余杭区****街道",
        latest_event="商家正在备货，预计 24 小时内发出",
    ),
    MockOrder(
        order_id="ORD202608230003",
        status="delivered",
        status_text="已签收",
        product_name="无线降噪耳机",
        quantity=2,
        amount=698.00,
        created_at="2026-08-15 11:20:00",
        paid_at="2026-08-15 11:20:48",
        shipped_at="2026-08-16 10:05:00",
        delivered_at="2026-08-18 16:42:00",
        carrier="中通快递",
        tracking_number="ZT MOCK 202608230003",
        receiver="陈*",
        receiver_phone="159****2468",
        receiver_address="北京市朝阳区****小区",
        latest_event="2026-08-18 16:42 已由本人签收",
    ),
)


class MockOrderRepository:
    """In-memory order repository that can later be replaced by a real API."""

    def __init__(self, orders: Iterable[MockOrder] | None = None):
        records = orders if orders is not None else DEFAULT_MOCK_ORDERS
        self._orders = {self.normalize_id(item.order_id): item for item in records}

    @staticmethod
    def normalize_id(order_id: str) -> str:
        return re.sub(r"\s+", "", order_id).strip().upper()

    def get(self, order_id: str) -> Dict[str, Any] | None:
        order = self._orders.get(self.normalize_id(order_id))
        return deepcopy(asdict(order)) if order else None

    def list_order_ids(self) -> List[str]:
        return list(self._orders)


class MockOrderQueryTool:
    """Detects order questions and queries the injected mock repository."""

    def __init__(self, repository: MockOrderRepository | None = None):
        self.repository = repository or MockOrderRepository()

    @staticmethod
    def is_order_query(query: str) -> bool:
        text = query.strip().lower()
        return bool(
            any(term in text for term in ORDER_QUERY_TERMS)
            or ORDER_ID_RE.search(query)
            or NUMERIC_ORDER_ID_RE.search(query)
        )

    @staticmethod
    def extract_order_id(query: str) -> str:
        match = ORDER_ID_RE.search(query) or NUMERIC_ORDER_ID_RE.search(query)
        return MockOrderRepository.normalize_id(match.group(0)) if match else ""

    def query(self, order_id: str) -> Dict[str, Any]:
        normalized_id = self.repository.normalize_id(order_id)
        order = self.repository.get(normalized_id)
        if order is None:
            return {
                "found": False,
                "order_id": normalized_id,
                "message": "未查询到该订单，请检查订单号是否正确。",
            }
        return {
            "found": True,
            "order_id": normalized_id,
            "order": order,
        }

    def query_from_text(self, query: str) -> Dict[str, Any] | None:
        if not self.is_order_query(query):
            return None
        order_id = self.extract_order_id(query)
        if not order_id:
            return {
                "found": False,
                "order_id": "",
                "missing_order_id": True,
                "message": "请提供订单号，例如 ORD202608230001，我可以为您查询订单状态和物流信息。",
            }
        return self.query(order_id)

    @staticmethod
    def format_answer(result: Dict[str, Any], language: str = "zh") -> str:
        if result.get("missing_order_id"):
            if language == "en":
                return "Please provide an order number, for example ORD202608230001."
            return str(result.get("message", ""))

        if not result.get("found"):
            order_id = result.get("order_id", "")
            if language == "en":
                return f"I couldn't find order {order_id}. Please check the order number."
            return f"未查询到订单 {order_id}，请检查订单号是否正确。"

        order = result["order"]
        tracking = order.get("tracking_number") or "暂无"
        carrier = order.get("carrier") or "暂无"
        if language == "en":
            return (
                f"Order {order['order_id']} is {order['status_text']}. "
                f"Item: {order['product_name']} × {order['quantity']}; "
                f"carrier: {carrier}; tracking number: {tracking}; "
                f"latest update: {order['latest_event']}."
            )
        return (
            f"订单 {order['order_id']} 当前状态：{order['status_text']}。\n"
            f"商品：{order['product_name']} × {order['quantity']}，实付 ¥{order['amount']:.2f}。\n"
            f"物流：{carrier}，运单号：{tracking}。\n"
            f"最新进度：{order['latest_event']}。"
        )
