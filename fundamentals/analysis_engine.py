from __future__ import annotations

from typing import Any, Optional

import numpy as np


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _fair_value_estimate(snapshot: dict[str, Any]) -> float:
    price = _safe_float(snapshot.get("current_price"))
    eps = _safe_float(snapshot.get("eps"))
    bvps = _safe_float(snapshot.get("bvps"))
    sales_ps = _safe_float(snapshot.get("sales_ps"))
    industry_pe = max(_safe_float(snapshot.get("industry_pe"), 18.0), 5.0)
    industry_pb = max(_safe_float(snapshot.get("industry_pb"), 2.5), 0.5)
    industry_ps = max(_safe_float(snapshot.get("industry_ps"), 2.0), 0.5)

    earnings_value = eps * industry_pe if eps > 0 else price
    book_value = bvps * industry_pb if bvps > 0 else price
    sales_value = sales_ps * industry_ps if sales_ps > 0 else price
    return float(np.mean([earnings_value, book_value, sales_value]))


def build_fundamental_report(snapshot: dict[str, Any], peers: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
    peers = peers or []
    price = _safe_float(snapshot.get("current_price"))
    pe = _safe_float(snapshot.get("pe"))
    pb = _safe_float(snapshot.get("pb"))
    roe = _safe_float(snapshot.get("roe"))
    debt_to_equity = _safe_float(snapshot.get("debt_to_equity"))
    revenue_growth = _safe_float(snapshot.get("revenue_growth"))
    earnings_growth = _safe_float(snapshot.get("earnings_growth"))
    operating_income = _safe_float(snapshot.get("operating_income"))
    net_income = _safe_float(snapshot.get("net_income"))
    assets = _safe_float(snapshot.get("assets"))
    liabilities = _safe_float(snapshot.get("liabilities"))
    capital_employed = max(assets - liabilities, 1.0)
    roce = (operating_income / capital_employed) * 100
    fair_value = _fair_value_estimate(snapshot)
    upside = ((fair_value / price) - 1) * 100 if price > 0 else 0.0

    profit_trend = "Improving" if earnings_growth > 8 else "Stable" if earnings_growth >= 0 else "Weakening"
    debt_health = "Healthy" if debt_to_equity < 0.6 else "Moderate" if debt_to_equity < 1.2 else "Stretched"

    peer_pe = np.mean([_safe_float(item.get("pe")) for item in peers]) if peers else _safe_float(snapshot.get("industry_pe"), 0.0)
    peer_pb = np.mean([_safe_float(item.get("pb")) for item in peers]) if peers else _safe_float(snapshot.get("industry_pb"), 0.0)
    sector_view = "Attractive" if pe <= max(peer_pe, 1.0) and pb <= max(peer_pb, 0.5) else "Premium"

    score = 50.0
    score += min(max(roe, 0.0), 25.0) * 0.9
    score += np.clip(revenue_growth, -10.0, 20.0) * 1.2
    score += np.clip(earnings_growth, -10.0, 20.0) * 1.4
    score += 10.0 if debt_health == "Healthy" else 0.0 if debt_health == "Moderate" else -10.0
    score += 10.0 if upside > 10 else 0.0
    score = float(np.clip(score, 0, 100))

    return {
        "pe": round(pe, 2),
        "pb": round(pb, 2),
        "roe": round(roe, 2),
        "roce": round(roce, 2),
        "revenue_growth": round(revenue_growth, 2),
        "profit_trend": profit_trend,
        "debt_health": debt_health,
        "sector_comparison": sector_view,
        "fair_value": round(fair_value, 2),
        "fair_value_upside_pct": round(upside, 2),
        "fundamental_score": round(score, 2),
        "net_income": round(net_income, 2),
    }
