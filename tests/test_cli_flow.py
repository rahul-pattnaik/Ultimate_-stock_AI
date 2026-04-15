from __future__ import annotations

import pandas as pd

import main_terminal


def test_cli_main_handles_valid_symbol_without_traceback(
    monkeypatch,
    capsys,
    bullish_df: pd.DataFrame,
    input_feeder,
    dummy_market_snapshot,
) -> None:
    import core.market_data_engine
    import intelligence.premium_news

    monkeypatch.setattr(main_terminal, "_select_symbols", lambda multiple=False: ["reliance"])
    monkeypatch.setattr(main_terminal, "_quiet_yfinance_download", lambda ticker: bullish_df if ticker == "RELIANCE.NS" else pd.DataFrame())
    monkeypatch.setattr(main_terminal, "get_stock_data", lambda symbol, verbose=True: bullish_df)
    monkeypatch.setattr(main_terminal, "_pause", lambda: None)
    monkeypatch.setattr(main_terminal, "_load_random_forest_signal", lambda: (_ for _ in ()).throw(ImportError("mocked")))
    monkeypatch.setattr(main_terminal, "_load_lstm_predictor", lambda: (_ for _ in ()).throw(ImportError("mocked")))
    monkeypatch.setattr(main_terminal, "_load_trading_agent", lambda: (_ for _ in ()).throw(ImportError("mocked")))
    monkeypatch.setattr(core.market_data_engine, "fetch_market_snapshot", lambda symbol: dummy_market_snapshot)
    monkeypatch.setattr(
        intelligence.premium_news,
        "news_sentiment_snapshot",
        lambda symbol: {"overall_sentiment": "Positive", "average_score": 1.0, "event_risk_alerts": [], "earnings_alert": "None"},
    )
    input_feeder(["1", "0"])
    main_terminal.main()
    output = capsys.readouterr().out
    assert "Deep Analysis - RELIANCE.NS" in output
    assert "Premium Intelligence Layer" in output
    assert "Traceback" not in output


def test_cli_main_handles_invalid_symbol_cleanly(monkeypatch, capsys, input_feeder) -> None:
    monkeypatch.setattr(main_terminal, "_select_symbols", lambda multiple=False: ["bad"])
    monkeypatch.setattr(main_terminal, "resolve_symbol", lambda raw: "BAD")

    def fake_get_stock_data(symbol: str, verbose: bool = True):
        print("[ERROR] No data for BAD. Possible reasons:")
        return None

    monkeypatch.setattr(main_terminal, "get_stock_data", fake_get_stock_data)
    monkeypatch.setattr(main_terminal, "_pause", lambda: None)
    input_feeder(["1", "0"])
    main_terminal.main()
    output = capsys.readouterr().out
    assert "[ERROR] No data for BAD" in output
    assert "Traceback" not in output


def test_cli_main_handles_empty_symbol_selection(monkeypatch, capsys, input_feeder) -> None:
    monkeypatch.setattr(main_terminal, "_select_symbols", lambda multiple=False: [])
    monkeypatch.setattr(main_terminal, "_pause", lambda: None)
    input_feeder(["1", "0"])
    main_terminal.main()
    output = capsys.readouterr().out
    assert "Unexpected error" not in output
    assert "Traceback" not in output


def test_cli_main_handles_no_data_returned(monkeypatch, capsys, input_feeder) -> None:
    monkeypatch.setattr(main_terminal, "_select_symbols", lambda multiple=False: ["RELIANCE"])
    monkeypatch.setattr(main_terminal, "_fetch_one", lambda raw_symbol, verbose=True: ("RELIANCE.NS", None))
    monkeypatch.setattr(main_terminal, "_pause", lambda: None)
    input_feeder(["1", "0"])
    main_terminal.main()
    output = capsys.readouterr().out
    assert "Traceback" not in output
