from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

import pandas as pd
import yfinance as yf


POSITIVE_WORDS = {"beat", "growth", "upgrade", "win", "strong", "record", "expands", "profit", "bullish"}
NEGATIVE_WORDS = {"miss", "downgrade", "weak", "fall", "loss", "probe", "lawsuit", "bearish", "delay"}
EVENT_WORDS = {"results", "earnings", "guidance", "merger", "stake", "pledge", "block", "buyback"}


def _quiet(fn, *args, **kwargs):
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        return fn(*args, **kwargs)


def _score_text(text: str) -> tuple[float, str]:
    words = set(text.lower().split())
    score = len(words & POSITIVE_WORDS) - len(words & NEGATIVE_WORDS)
    label = "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"
    return score, label


def news_sentiment_snapshot(symbol: str) -> dict[str, Any]:
    try:
        ticker = yf.Ticker(symbol)
        news_items = _quiet(lambda: ticker.news) or []
    except Exception:
        news_items = []

    parsed = []
    event_alerts = []
    for item in news_items[:8]:
        title = str(item.get("title") or item.get("content", {}).get("title") or "")
        publisher = str(item.get("publisher") or item.get("content", {}).get("provider", {}).get("displayName") or "Unknown")
        score, label = _score_text(title)
        parsed.append({"title": title, "publisher": publisher, "sentiment": label, "score": score})
        if any(word in title.lower() for word in EVENT_WORDS):
            event_alerts.append(title)

    average = sum(item["score"] for item in parsed) / len(parsed) if parsed else 0.0
    if average > 0.2:
        overall = "Positive"
    elif average < -0.2:
        overall = "Negative"
    else:
        overall = "Neutral"

    earnings_alert = None
    try:
        calendar = _quiet(lambda: ticker.calendar)
        if isinstance(calendar, pd.DataFrame) and not calendar.empty:
            earnings_alert = "Upcoming earnings event detected"
    except Exception:
        earnings_alert = None

    return {
        "overall_sentiment": overall,
        "average_score": round(average, 2),
        "latest_news": parsed,
        "event_risk_alerts": event_alerts[:5],
        "earnings_alert": earnings_alert or "No earnings calendar available",
        "graceful_fallback": not bool(parsed),
    }
