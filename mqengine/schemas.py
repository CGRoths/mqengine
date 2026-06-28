from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import pandas as pd


@dataclass(slots=True)
class ValidationIssue:
    code: str
    message: str
    severity: str = "error"
    count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ValidationReport:
    errors: list[ValidationIssue]
    warnings: list[ValidationIssue]

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
        }

    def raise_if_errors(self) -> None:
        if self.errors:
            summary = "; ".join(f"{issue.code}: {issue.message}" for issue in self.errors)
            raise ValueError(summary)


@dataclass(slots=True)
class OrderEvent:
    ts: pd.Timestamp
    order_id: str
    symbol: str
    side: str
    quantity: float
    status: str
    venue: str | None = None
    limit_price: float | None = None
    created_ts: pd.Timestamp | None = None
    updated_ts: pd.Timestamp | None = None


@dataclass(slots=True)
class FillEvent:
    ts: pd.Timestamp
    order_id: str
    symbol: str
    side: str
    quantity: float
    fill_price: float
    fee: float = 0.0
    venue: str | None = None
    arrival_price: float | None = None
    latency_ms: float | None = None


@dataclass(slots=True)
class PositionSnapshot:
    ts: pd.Timestamp
    symbol: str
    quantity: float
    mark_price: float
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass(slots=True)
class TargetPositionSnapshot:
    ts: pd.Timestamp
    symbol: str
    target_quantity: float
    target_weight: float | None = None


@dataclass(slots=True)
class EquitySnapshot:
    ts: pd.Timestamp
    equity: float
    cash: float | None = None
    gross_exposure: float | None = None
    net_exposure: float | None = None
