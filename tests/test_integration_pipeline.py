from __future__ import annotations

import pandas as pd

from ai.premium_scoring import generate_premium_scores
from ai.ranking_engine import stock_score
from fundamentals.analysis_engine import build_fundamental_report
from main_terminal import _infer_fundamental_inputs
from technical.advanced_analysis import advanced_technical_report


def test_fundamentals_technicals_and_scores_pipeline(bullish_df: pd.DataFrame) -> None:
    inferred = _infer_fundamental_inputs("RELIANCE.NS", bullish_df)
    technical_report = advanced_technical_report(bullish_df)
    fundamental_report = build_fundamental_report(inferred)
    scores = generate_premium_scores(technical_report, fundamental_report, ai_signal_result={"score": stock_score(bullish_df)})

    assert "trend" in technical_report
    assert "fair_value" in fundamental_report
    assert 0 <= scores["buy_score"] <= 100


def test_symbol_fetch_indicator_report_flow(monkeypatch, bullish_df: pd.DataFrame) -> None:
    import main_terminal

    monkeypatch.setattr(main_terminal, "resolve_symbol", lambda raw: "RELIANCE.NS")
    monkeypatch.setattr(main_terminal, "get_stock_data", lambda symbol, verbose=True: bullish_df)
    symbol, df = main_terminal._fetch_one("reliance")
    assert symbol == "RELIANCE.NS"
    assert df is not None
    report = advanced_technical_report(df)
    assert "confluence_score" in report
