# ai/sentiment_model.py
# ─────────────────────────────────────────────────────────────────────────────
# News Sentiment Analyzer
# Primary  : FinBERT (finance-specific BERT — far more accurate than generic)
# Fallback : VADER (lightweight, no GPU required)
# Output   : label, numeric score (-1 to +1), confidence
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import re


# ── FinBERT (preferred) ───────────────────────────────────────────────────────

def _load_finbert():
    """Load FinBERT lazily — only when first called."""
    try:
        from transformers import pipeline
        # ProsusAI/finbert is trained specifically on financial news
        nlp = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            top_k=None,           # return all class scores
        )
        return nlp, "finbert"
    except Exception:
        return None, None


def _finbert_score(nlp, text: str) -> dict:
    """Run FinBERT and return normalised result."""
    # Truncate to 512 tokens to avoid overflow
    text    = text[:1200]
    results = nlp(text)[0]            # list of {label, score}
    scores  = {r["label"].lower(): r["score"] for r in results}

    positive = scores.get("positive", 0.0)
    negative = scores.get("negative", 0.0)
    neutral  = scores.get("neutral",  0.0)

    # Composite score: +1 fully positive, -1 fully negative
    composite = positive - negative
    confidence = max(positive, negative, neutral)

    label = (
        "POSITIVE" if composite > 0.15 else
        "NEGATIVE" if composite < -0.15 else
        "NEUTRAL"
    )
    return {
        "label":      label,
        "score":      round(composite, 4),
        "confidence": round(confidence, 4),
        "breakdown":  {k: round(v, 4) for k, v in scores.items()},
        "model":      "FinBERT",
    }


# ── VADER fallback ────────────────────────────────────────────────────────────

def _vader_score(text: str) -> dict:
    """Lightweight rule-based sentiment (no model download needed)."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sia    = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores(text)
        comp   = scores["compound"]
        label  = "POSITIVE" if comp > 0.05 else "NEGATIVE" if comp < -0.05 else "NEUTRAL"
        return {
            "label":      label,
            "score":      round(comp, 4),
            "confidence": round(max(scores["pos"], scores["neg"], scores["neu"]), 4),
            "breakdown":  {k: round(v, 4) for k, v in scores.items()},
            "model":      "VADER",
        }
    except ImportError:
        return _keyword_score(text)


def _keyword_score(text: str) -> dict:
    """Minimal keyword-based fallback — no dependencies."""
    text_l = text.lower()
    bull   = ["beat", "profit", "growth", "surge", "rally", "strong",
              "upgrade", "buy", "revenue", "record", "positive", "rise"]
    bear   = ["miss", "loss", "decline", "fall", "weak", "downgrade",
              "sell", "risk", "concern", "drop", "negative", "cut"]
    pos = sum(w in text_l for w in bull)
    neg = sum(w in text_l for w in bear)
    total = pos + neg or 1
    comp  = (pos - neg) / total
    label = "POSITIVE" if comp > 0.1 else "NEGATIVE" if comp < -0.1 else "NEUTRAL"
    return {
        "label":      label,
        "score":      round(comp, 4),
        "confidence": round(abs(comp), 4),
        "breakdown":  {"positive_words": pos, "negative_words": neg},
        "model":      "Keyword",
    }


# ── Public API ────────────────────────────────────────────────────────────────

_finbert_nlp  = None
_finbert_name = None


def analyze_news(text: str) -> dict:
    """
    Analyze sentiment of a single news headline or article snippet.

    Args:
        text : any financial news string

    Returns dict:
        label      : "POSITIVE" | "NEGATIVE" | "NEUTRAL"
        score      : float in [-1, +1]
        confidence : float in [0, 1]
        breakdown  : per-class probabilities
        model      : which model was used
    """
    global _finbert_nlp, _finbert_name

    if not text or not text.strip():
        return {"error": "Empty text provided"}

    # Try FinBERT first (load once, reuse)
    if _finbert_nlp is None:
        _finbert_nlp, _finbert_name = _load_finbert()

    if _finbert_nlp:
        return _finbert_score(_finbert_nlp, text)
    else:
        # Graceful degradation
        return _vader_score(text)


def analyze_batch(texts: list[str]) -> list[dict]:
    """
    Analyze a list of headlines. Returns aggregated summary + per-item results.
    """
    results  = [analyze_news(t) for t in texts]
    scores   = [r["score"] for r in results if "score" in r]
    avg      = sum(scores) / len(scores) if scores else 0.0
    positive = sum(1 for s in scores if s > 0.15)
    negative = sum(1 for s in scores if s < -0.15)
    neutral  = len(scores) - positive - negative

    overall = (
        "POSITIVE 📈" if avg > 0.1 else
        "NEGATIVE 📉" if avg < -0.1 else
        "NEUTRAL ➡️"
    )

    return {
        "overall_label":    overall,
        "average_score":    round(avg, 4),
        "positive_count":   positive,
        "negative_count":   negative,
        "neutral_count":    neutral,
        "individual":       results,
    }
