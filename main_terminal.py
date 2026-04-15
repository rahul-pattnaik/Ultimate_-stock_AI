import io
import os
import re
import sys
from contextlib import redirect_stderr, redirect_stdout
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.backtester import Backtester
from core.config import NIFTY50_STOCKS, STOCKS, US_STOCKS
from core.data_fetcher import get_latest_price, get_multiple_stocks, get_stock_data
from core.nse_universe import (
    get_custom_watchlist,
    get_nifty50,
    get_nifty500,
    get_sector,
    search_symbol,
)
from core.utils import fmt_price, fmt_volume
from portfolio.sharpe_ratio import sharpe_ratio
from portfolio.sortino_ratio import sortino_ratio

if TYPE_CHECKING:
    from fundamentals.financials import FinancialDataHandler


MENU = """
==============================================================================
 Ultimate Stock AI - Integrated Terminal
==============================================================================
 1. Deep Analysis
 2. Scanner Suite
 3. Fundamentals and Valuation
 4. AI Lab
 5. Portfolio Analytics
 6. Strategy Backtest
 7. Market Intelligence
 8. Options Tools
 9. Watchlists and Symbol Tools
10. Repo Review
11. Premium Command Center
 0. Exit
==============================================================================
"""


PROMPT_NOTES = {
    "Choose feature": "Enter the menu number for the feature you want to run.",
    "Choose universe": "Pick where symbols should come from. Use 1 for your own stock names.",
    "Symbols separated by commas": "Enter one or more stock symbols separated by commas, for example: ITC, RELIANCE, TCS.",
    "Sector (it, bank, pharma, auto, nifty50, nifty500)": "Enter one supported sector keyword exactly as shown.",
    "Search text": "Enter part of a stock symbol or company name to search.",
    "Select result": "Enter the serial number of the matching stock you want to use.",
    "Sector label": "Enter a short sector name for peer comparison grouping, for example IT or Banking.",
    "Industry label": "Enter a short industry name for peer comparison, for example Software or FMCG.",
    "Choose AI feature": "Enter the AI menu number you want to run, or 7 for the full AI suite.",
    "Use batch headlines": "Choose yes to analyze multiple headlines together, or no for one headline.",
    "Headlines separated by |": "Enter multiple headlines separated by the | symbol.",
    "Headline text": "Enter one news headline or short article text for sentiment analysis.",
    "Episodes": "Enter how many RL training passes to run. Higher values are slower.",
    "Choose strategy": "Enter the backtest strategy number you want to test.",
    "Initial capital": "Enter the starting portfolio money for the backtest, for example 100000 or 10000.",
    "Position size fraction": "Enter the fraction of capital used per trade. Example: 0.1 means 10% of capital each trade.",
    "Stop loss fraction": "Enter the loss limit per trade as a decimal. Example: 0.05 means exit after a 5% loss.",
    "Take profit fraction": "Enter the profit target per trade as a decimal. Example: 0.15 means exit after a 15% gain.",
    "Headline": "Enter a market or company headline to evaluate sentiment and intelligence signals.",
    "Insider buy amount": "Enter the total insider buying amount used by the simple insider signal model.",
    "Insider sell amount": "Enter the total insider selling amount used by the simple insider signal model.",
    "Promoter pledge %": "Enter the promoter pledge percentage. Higher values mean higher risk.",
    "Use live volume for block-deal check": "Choose yes to use fetched market data volume, or no to enter volume manually.",
    "Current volume": "Enter the latest traded volume number for the stock.",
    "Average volume": "Enter the normal average traded volume used for comparison.",
    "Underlying price": "Enter the current stock price to build the sample options spread.",
    "Run NSE search": "Choose yes if you want to search NSE symbols interactively.",
    "Resolve a bare symbol": "Choose yes to convert a raw symbol like ITC into a market ticker like ITC.NS.",
    "Bare symbol": "Enter the stock symbol without exchange suffix, for example ITC or RELIANCE.",
    "Monthly SIP amount": "Enter the monthly investment amount you want to simulate.",
    "Expected annual return %": "Enter the expected annual portfolio return in percent for SIP simulation.",
    "SIP years": "Enter the SIP duration in years.",
    "Export base filename": "Enter a short filename stem for CSV and PDF exports.",
    "Export premium report": "Choose yes to save the premium report in both CSV and PDF formats.",
}


PRODUCTION_READY = [
    "core.data_fetcher",
    "core.backtester",
    "core.nse_universe (fallbacks especially)",
    "technical.trend_detection",
    "technical.breakout",
    "technical.support_resistance",
    "technical.volume_profile",
    "technical.momentum",
    "technical.moving_averages",
    "technical.supertrend",
    "technical.volatility",
    "technical.vwap",
    "technical.fibonacci",
    "technical.ichimoku",
    "technical.elliott_wave",
    "ai.signal_model",
    "ai.price_prediction",
    "ai.ranking_engine",
    "portfolio.optimizer",
    "portfolio.sharpe_ratio",
    "portfolio.sortino_ratio",
]

BETA_READY = [
    "scanners.breakout_scanner",
    "scanners.momentum_scanner",
    "scanners.swing_scanner",
    "fundamentals.financials",
    "fundamentals.ratios",
    "fundamentals.valuation",
    "fundamentals.peer_comparison",
    "scanners.value_scanner",
]

EXPERIMENTAL = [
    "ai.random_forest_signal",
    "ai.lstm_predictor",
    "ai.reinforcement_agent",
    "ai.sentiment_model (best effort if extras are missing)",
    "intelligence.block_deals",
    "intelligence.insider_trading",
    "intelligence.promoter_pledge",
    "intelligence.fii_dii",
    "options.options_strategy_builder",
]


def _setup_console() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _line(char: str = "=", width: int = 78) -> None:
    print(char * width)


def _section(title: str) -> None:
    print()
    _line()
    print(title)
    _line()


def _subsection(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def _pause() -> None:
    print("Note: Press Enter to return to the main menu.")
    input("\nPress Enter to continue...")


def _prompt_note(text: str) -> Optional[str]:
    normalized = re.sub(r"\s+\([^)]+\)$", "", text).strip()
    return PROMPT_NOTES.get(text) or PROMPT_NOTES.get(normalized)


def _ask(text: str, default: str = "", note: Optional[str] = None) -> str:
    if note is None:
        note = _prompt_note(text)
    if note:
        print(f"Note: {note}")
    elif default:
        print("Note: Enter a value or press Enter to use the default shown in brackets.")
    else:
        print("Note: Enter the requested value and press Enter.")
    prompt = text
    if default:
        prompt += f" [{default}]"
    prompt += ": "
    raw = input(prompt).strip()
    return raw or default


def _ask_int(text: str, default: int) -> int:
    note = _prompt_note(text) or "Enter a whole number. Press Enter to keep the default value."
    try:
        return int(_ask(text, str(default), note=note))
    except ValueError:
        return default


def _ask_float(text: str, default: float) -> float:
    note = _prompt_note(text) or "Enter a decimal number. Press Enter to keep the default value."
    try:
        return float(_ask(text, str(default), note=note))
    except ValueError:
        return default


def _ask_yes_no(text: str, default: bool = True) -> bool:
    default_label = "Y/n" if default else "y/N"
    note = _prompt_note(text) or "Type y for yes or n for no. Press Enter to use the default option."
    raw = _ask(f"{text} ({default_label})", "", note=note).lower()
    if not raw:
        return default
    return raw in {"y", "yes", "1", "true"}


def _safe(fn: Callable, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        return {"error": str(exc)}


def _print_mapping(mapping: dict, preferred: Optional[list[str]] = None) -> None:
    keys = preferred or list(mapping.keys())
    for key in keys:
        if key in mapping:
            print(f"{key:<24} {mapping[key]}")


def _print_list(items: list[str], limit: Optional[int] = None) -> None:
    use_items = items if limit is None else items[:limit]
    for item in use_items:
        print(f"- {item}")


def _clip_score(score: float) -> float:
    return float(np.clip(score, 0, 100))


def _average_score(*scores: Optional[float]) -> float:
    values = [float(score) for score in scores if score is not None]
    if not values:
        return 50.0
    return _clip_score(sum(values) / len(values))


def _signal_score(signal: object, default: float = 50.0) -> float:
    text = str(signal).upper()
    if "STRONG BUY" in text:
        return 92.0
    if re.search(r"\bBUY\b", text):
        return 74.0
    if "STRONG SELL" in text:
        return 8.0
    if re.search(r"\bSELL\b", text):
        return 26.0
    if "HOLD" in text or "NEUTRAL" in text or "SIDEWAYS" in text:
        return 50.0
    if "BULLISH" in text or "UPTREND" in text or "BREAKOUT" in text or "POSITIVE" in text:
        return 68.0
    if "BEARISH" in text or "DOWNTREND" in text or "BREAKDOWN" in text or "NEGATIVE" in text:
        return 32.0
    if "LOW RISK" in text:
        return 66.0
    if "MEDIUM RISK" in text:
        return 42.0
    if "HIGH RISK" in text:
        return 18.0
    if "POSSIBLE BLOCK DEAL" in text:
        return 44.0
    if "NORMAL" in text:
        return 52.0
    if "UNAVAILABLE" in text or "ERROR" in text:
        return default
    return default


def _rsi_score_value(rsi: float) -> float:
    if rsi < 30:
        return 85.0
    if rsi < 45:
        return 68.0
    if rsi < 60:
        return 55.0
    if rsi < 70:
        return 40.0
    return 18.0


def _pct_score(percent: float, scale: float = 1.5) -> float:
    return _clip_score(50 + percent * scale)


def _upside_score(percent: float) -> float:
    return _clip_score(50 + percent * 1.2)


def _extract_confidence(value: object) -> Optional[float]:
    if isinstance(value, dict):
        confidence = value.get("confidence")
    else:
        confidence = getattr(value, "confidence", None)

    if isinstance(confidence, str):
        match = re.search(r"-?\d+(\.\d+)?", confidence)
        return float(match.group()) if match else None
    if isinstance(confidence, (int, float, np.floating)):
        return float(confidence)
    return None


def _prediction_score(prediction: object) -> Optional[float]:
    if not isinstance(prediction, dict) or "error" in prediction:
        return None
    change = prediction.get("change_percent")
    confidence = _extract_confidence(prediction)
    if isinstance(change, (int, float, np.floating)):
        return _average_score(_upside_score(float(change)), confidence)
    return confidence


def _print_verdict(label: str, score: float) -> None:
    score = _clip_score(score)
    if score >= 80:
        verdict = "STRONG BUY"
    elif score >= 62:
        verdict = "BUY"
    elif score >= 45:
        verdict = "HOLD"
    elif score >= 25:
        verdict = "SELL"
    else:
        verdict = "STRONG SELL"
    print(f"Verdict - {label:<16} {verdict} ({score:.1f}/100)")


def _verdict_label(score: float) -> str:
    score = _clip_score(score)
    if score >= 80:
        return "STRONG BUY"
    if score >= 62:
        return "BUY"
    if score >= 45:
        return "HOLD"
    if score >= 25:
        return "SELL"
    return "STRONG SELL"


def _num(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
        if match:
            return float(match.group())
    return default


def _pct_text(value: float, digits: int = 2) -> str:
    return f"{value:+.{digits}f}%"


def _plain_signal(value: object) -> str:
    return re.sub(r"[^\x00-\x7F]+", "", str(value)).strip()


def _print_note_lines(lines: list[str]) -> None:
    for line in lines:
        print(line)


def _progress(message: str) -> None:
    print(f"[RUNNING] {message}", flush=True)


def _choose_from_symbols(symbols: list[str], title: str = "Select Symbol") -> list[str]:
    if not symbols:
        return []
    if len(symbols) == 1:
        return symbols
    _section(title)
    print("Multiple symbols are available for this feature. Choose one stock to analyze.")
    for idx, symbol in enumerate(symbols, start=1):
        print(f"{idx}. {symbol}")
    pick = max(1, min(_ask_int("Select result", 1), len(symbols)))
    return [symbols[pick - 1]]


def _print_deep_analysis_report(
    symbol: str,
    df: pd.DataFrame,
    handler: "FinancialDataHandler",
    inferred: dict,
    technical_score: float,
    ai_layer_score: float,
    fundamentals_score: float,
    composite_score: float,
    trend: object,
    trend_detail: object,
    breakout: object,
    support_resistance: object,
    sr_zones: object,
    volume_poc: object,
    volume_profile_result: object,
    momentum: object,
    moving_averages: object,
    supertrend: object,
    volatility: object,
    vwap: object,
    anchored_vwap_tail: object,
    fibonacci: object,
    ichimoku: object,
    elliott_wave: object,
    prediction: object,
    ai_signal_result: object,
    ai_score_result: object,
    ai_score_detail: object,
    rf_result: object,
    lstm_result: object,
    rl_result: object,
    value_result: object,
    block_deal_signal: object,
    insider: object,
    pledge: object,
) -> None:
    overall_score = _average_score(technical_score, ai_layer_score, fundamentals_score, composite_score)
    current_price = float(df["Close"].iloc[-1])
    volatility_pct = handler.get_volatility(symbol)
    momentum_20d = handler.get_price_momentum(symbol).get("20d", 0.0)
    trend_detail = trend_detail if isinstance(trend_detail, dict) else {}
    breakout = breakout if isinstance(breakout, dict) else {}
    momentum = momentum if isinstance(momentum, dict) else {}
    moving_averages = moving_averages if isinstance(moving_averages, dict) else {}
    supertrend = supertrend if isinstance(supertrend, dict) else {}
    volatility = volatility if isinstance(volatility, dict) else {}
    vwap = vwap if isinstance(vwap, dict) else {}
    fibonacci = fibonacci if isinstance(fibonacci, dict) else {}
    ichimoku = ichimoku if isinstance(ichimoku, dict) else {}
    elliott_wave = elliott_wave if isinstance(elliott_wave, dict) else {}
    ai_signal_result = ai_signal_result if isinstance(ai_signal_result, dict) else {}
    ai_score_detail = ai_score_detail if isinstance(ai_score_detail, dict) else {}
    prediction = prediction if isinstance(prediction, dict) else {}
    rf_result = rf_result if isinstance(rf_result, dict) else {"signal": str(rf_result)}
    lstm_result = lstm_result if isinstance(lstm_result, dict) else {"signal": str(lstm_result)}
    rl_result = rl_result if isinstance(rl_result, dict) else {"signal": str(rl_result)}
    value_dict = value_result.to_dict() if hasattr(value_result, "to_dict") else (value_result if isinstance(value_result, dict) else {})
    sr_zone_support = None
    sr_zone_resistance = None
    risk_reward = None
    if isinstance(sr_zones, dict):
        sr_zone_support = sr_zones.get("nearest_support")
        sr_zone_resistance = sr_zones.get("nearest_resistance")
        risk_reward = sr_zones.get("risk_reward")

    print()
    _line("=")
    print(f"Stock: {symbol}")
    print(f"Overall Verdict: {_verdict_label(overall_score)} ({overall_score:.1f}/100)")
    _line("=")
    print("Meaning:")
    overall_notes = []
    if overall_score >= 62:
        overall_notes.append("Clear positive bias, but still not risk-free.")
    elif overall_score >= 45:
        overall_notes.extend([
            "Not a clear strong buy",
            "Not a panic sell",
            "Mixed signals",
            "Better for watchlist / partial position / wait for confirmation",
        ])
    else:
        overall_notes.extend([
            "Weak setup right now",
            "Risk is elevated versus reward",
            "Better to wait for stronger confirmation",
        ])
    _print_note_lines(overall_notes)

    print("\nWhy mixed?")
    mixed_notes = []
    if momentum_20d > 0:
        mixed_notes.append("Short-term price action improved")
    else:
        mixed_notes.append("Short-term price action is still weak")
    timeframes = trend_detail.get("timeframes", {})
    if any("Bearish" in str(value) for value in timeframes.values()):
        mixed_notes.append("Medium/long trend still weak")
    if technical_score > fundamentals_score:
        mixed_notes.append("Technicals better than fundamentals")
    if abs(technical_score - ai_layer_score) >= 10 or abs(ai_layer_score - fundamentals_score) >= 10:
        mixed_notes.append("AI models disagree with the other layers")
    if volatility_pct >= 30:
        mixed_notes.append("Volatility is high (risky moves)")
    _print_note_lines(mixed_notes or ["Signals are broadly aligned."])

    _section("1. Basic Snapshot")
    print(f"Current Price {fmt_price(current_price)}")
    print("Current market price.")
    print()
    print(f"Bars {len(df)}")
    print("Trading days of data used for this report.")
    print()
    print(f"Date Range {df.index[0].date()} to {df.index[-1].date()}")
    print("Historical period analyzed.")
    print()
    print(f"Volatility {volatility_pct:.2f}%")
    if volatility_pct >= 30:
        _print_note_lines([
            "Stock moves a lot. Higher than many stable stocks.",
            "Meaning:",
            "Good for traders",
            "Risky for conservative investors",
        ])
    else:
        print("Volatility is manageable relative to a high-beta trading stock.")
    print()
    print(f"Momentum 20d = {_pct_text(momentum_20d)}")
    print("Recent 20-day performance. Positive means short-term strength.")

    _section("2. Technical Layer")
    print(f"Trend = {trend}")
    if timeframes:
        print("Timeframe view:")
        for label, value in timeframes.items():
            print(f"{label.replace('_', ' ').title()} = {value}")
    print("Meaning:")
    if "Downtrend" in str(trend):
        print("Recent bounce exists, but the larger trend structure is still weak.")
    elif "Uptrend" in str(trend):
        print("Trend structure is supportive across most timeframes.")
    else:
        print("Trend is mixed and needs more confirmation.")
    print()

    ma_values = trend_detail.get("moving_averages", {})
    price_vs_ma = trend_detail.get("price_vs_ma", {})
    print("Moving Averages Explained")
    print(f"Price = {fmt_price(current_price)}")
    if ma_values:
        if "vs_MA20" in price_vs_ma:
            print(f"Above MA20 = {price_vs_ma['vs_MA20']}")
        if "vs_MA50" in price_vs_ma:
            print(f"Above MA50 = {price_vs_ma['vs_MA50']}")
        if "vs_MA200" in price_vs_ma:
            print(f"Vs MA200 = {price_vs_ma['vs_MA200']}")
    print("This is one reason signals can conflict across short and long timeframes.")
    print()

    print(f"Breakout = {breakout.get('signal', breakout)}")
    triggered_signals = breakout.get("signals_hit", [])
    if triggered_signals:
        print("Why?")
        _print_note_lines(triggered_signals[:4])
    print("Meaning:")
    if breakout.get("triggered"):
        print("Breakout is already confirmed by this model.")
    else:
        print("Possible breakout is forming, but confirmation is still needed.")
    print()

    print("Support / Resistance")
    if sr_zone_support is not None:
        print(f"Nearest Support: {fmt_price(float(sr_zone_support))}")
    elif isinstance(support_resistance, tuple) and len(support_resistance) == 2:
        print(f"Support Zone: {fmt_price(float(support_resistance[0]))}")
    if sr_zone_resistance is not None:
        print(f"Nearest Resistance: {fmt_price(float(sr_zone_resistance))}")
    elif isinstance(support_resistance, tuple) and len(support_resistance) == 2:
        print(f"Resistance Zone: {fmt_price(float(support_resistance[1]))}")
    if risk_reward is not None:
        print(f"Risk Reward = {_num(risk_reward):.2f}")
    print("Meaning:")
    print("Support is where buyers may defend price. Resistance is where sellers may appear.")
    print()

    print("Volume Profile")
    print(f"POC = {fmt_price(_num(volume_poc))}")
    if isinstance(volume_profile_result, dict):
        print(f"Price Position = {volume_profile_result.get('price_position', 'N/A')}")
    print("Meaning:")
    print("POC is the price area with the most historical participation and often acts like a magnet zone.")
    print()

    print(f"Momentum Section = {momentum.get('signal', momentum)}")
    indicators_block = momentum.get("indicators", {})
    if indicators_block:
        print(f"RSI = {_num(indicators_block.get('RSI_14')):.2f}")
        print(f"Stochastic = {_num(indicators_block.get('Stoch_K')):.2f}")
        print(f"CCI = {_num(indicators_block.get('CCI_20')):.2f}")
    print("Meaning:")
    print("Momentum indicators show whether price is accelerating cleanly or getting stretched.")
    print()

    print(f"Moving Averages Verdict = {moving_averages.get('signal', moving_averages)}")
    if "bull_count" in moving_averages:
        print(moving_averages["bull_count"])
    print()
    print(f"Supertrend = {supertrend.get('direction', supertrend)}")
    recent_flips = supertrend.get("recent_flips", [])
    if recent_flips:
        latest_flip = recent_flips[-1]
        print(f"Recent flip: {latest_flip.get('direction')} on {latest_flip.get('date')}")
    print()
    print(f"Volatility = {volatility.get('volatility_regime', volatility)}")
    atr_pct = volatility.get("ATR_pct")
    if atr_pct is not None:
        print(f"ATR % = {atr_pct}")
    print()
    print("VWAP Bands")
    print(f"Position = {vwap.get('position', vwap)}")
    print()
    print("Fibonacci Levels")
    if "nearest_fib_support" in fibonacci:
        print(f"Nearest support = {fmt_price(_num(fibonacci['nearest_fib_support']))}")
    if "nearest_fib_resistance" in fibonacci:
        print(f"Nearest resistance = {fmt_price(_num(fibonacci['nearest_fib_resistance']))}")
    print()
    print(f"Ichimoku = {ichimoku.get('signal', ichimoku)}")
    print()
    print("Elliott Wave")
    print(f"Structure = {elliott_wave.get('current_wave', elliott_wave)}")
    fib_targets = elliott_wave.get("fibonacci_targets", {})
    if fib_targets:
        print("Target zones:")
        for value in list(fib_targets.values())[:3]:
            print(fmt_price(_num(value)))
    print()
    print(f"Technical Final Verdict = {_verdict_label(technical_score)} ({technical_score:.1f}/100)")

    _section("3. AI Layer")
    if "error" in prediction:
        print(f"AI Prediction Error = {prediction['error']}")
        print("This means one prediction model failed and needs fixing.")
    else:
        print(f"AI Prediction = {prediction}")
    print()
    print(f"AI Signal = {ai_signal_result.get('signal', ai_signal_result)}")
    if "score" in ai_signal_result:
        print(f"Score = {_num(ai_signal_result.get('score')):.0f}")
    reasons = ai_signal_result.get("reasons", [])
    if reasons:
        print("Why?")
        _print_note_lines(reasons[:5])
    print()
    print(f"AI Score = {ai_score_result}")
    if isinstance(ai_score_detail, dict) and "grade" in ai_score_detail:
        print(f"Grade = {ai_score_detail['grade']}")
    print()
    print(f"Random Forest = {rf_result.get('signal', rf_result)}")
    training = rf_result.get("training", {})
    if training:
        print(f"CV accuracy = {_num(training.get('cv_accuracy_mean')) * 100:.1f}%")
    print()
    lstm_direction = lstm_result.get("direction", lstm_result.get("signal", lstm_result))
    print(f"LSTM Predictor = {lstm_direction}")
    if "predicted_prices" in lstm_result:
        print(f"Predicted next prices = {lstm_result['predicted_prices']}")
    print()
    print("RL Model")
    if "win_rate" in rl_result:
        print(f"Win rate = {rl_result['win_rate']}")
    if "total_pnl" in rl_result:
        print(f"Total PnL = {rl_result['total_pnl']}")
    print()
    print(f"AI Final Verdict = {_verdict_label(ai_layer_score)} ({ai_layer_score:.1f}/100)")

    _section("4. Fundamentals Layer")
    print("Ratios")
    print(f"PE = {_num(inferred.get('pe')):.2f}")
    print(f"PB = {_num(inferred.get('pb')):.2f}")
    print(f"PS = {_num(inferred.get('ps')):.2f}")
    print()
    grade = value_dict.get("recommendation", value_dict.get("grade", getattr(value_result, "grade", "N/A")))
    print(f"Value Scanner Grade = {getattr(value_result, 'grade', value_dict.get('grade', 'N/A'))}")
    print("Positive Fundamentals")
    positive_fundamentals = []
    if "Bullish" in str(insider):
        positive_fundamentals.append("Insider signal bullish")
    if "Low Risk" in str(pledge):
        positive_fundamentals.append("Low pledge risk")
    if _num(inferred.get("debt_to_equity")) < 0.7:
        positive_fundamentals.append("Conservative leverage")
    if _num(inferred.get("current_ratio")) > 1.0:
        positive_fundamentals.append("Current ratio decent")
    if _num(inferred.get("roe")) > 8:
        positive_fundamentals.append("Return on equity acceptable")
    _print_note_lines(positive_fundamentals or ["Balance-sheet quality is acceptable."])
    print("Negative Fundamentals")
    negative_fundamentals = []
    if _num(inferred.get("revenue_growth")) <= 0:
        negative_fundamentals.append("Revenue growth weak or negative")
    if _num(inferred.get("earnings_growth")) < 5:
        negative_fundamentals.append("Growth score is not strong")
    if getattr(value_result, "grade", "") in {"C", "D"}:
        negative_fundamentals.append("Valuation is not a screaming bargain")
    _print_note_lines(negative_fundamentals or ["No major red flag in the simple snapshot inputs."])
    print()
    print(f"Fundamentals Verdict = {_verdict_label(fundamentals_score)} ({fundamentals_score:.1f}/100)")

    _section("5. Composite Final Verdict")
    print(f"Technical = {_verdict_label(technical_score)} ({technical_score:.1f})")
    print(f"AI = {_verdict_label(ai_layer_score)} ({ai_layer_score:.1f})")
    print(f"Fundamentals = {_verdict_label(fundamentals_score)} ({fundamentals_score:.1f})")
    print(f"Composite = {_verdict_label(composite_score)} ({composite_score:.1f})")
    print(f"Overall = {_verdict_label(overall_score)} ({overall_score:.1f})")

    print("\nWhat This Means")
    if overall_score >= 62:
        _print_note_lines([
            "Momentum is supportive.",
            "Trend is improving enough to justify a bullish stance.",
            "Risk still needs monitoring.",
        ])
    elif overall_score >= 45:
        _print_note_lines([
            "Recovering stock, but not yet fully strong.",
            "Good for swing traders and breakout watchers.",
            "Less ideal for safe long-term fresh lump-sum buying.",
        ])
    else:
        _print_note_lines([
            "Setup is weak right now.",
            "Better to wait for trend repair or better value.",
        ])

    print("\nBiggest Red Flags")
    red_flags = []
    adx_value = _num(trend_detail.get("adx"), 0.0)
    if adx_value > 100:
        red_flags.append(f"ADX = {adx_value:.2f} looks unrealistic and may be a coding issue.")
    if getattr(value_result, "intrinsic_value", 0) and _num(getattr(value_result, "intrinsic_value", 0)) > current_price * 100:
        red_flags.append("Intrinsic value looks unrealistically large, so the DCF assumptions need review.")
    if isinstance(prediction, dict) and "error" in prediction:
        red_flags.append("AI prediction hit a shape mismatch bug.")
    if _num(rl_result.get("win_rate"), 100) < 40:
        red_flags.append("RL model performance is poor and should not drive decisions.")
    if not red_flags:
        red_flags.append("No obvious output-quality red flag was detected in this run.")
    _print_note_lines(red_flags)


def _lowercase_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    lowered = df.copy()
    lowered.columns = [str(col).lower() for col in lowered.columns]
    return lowered


def _quiet_yfinance_download(ticker: str) -> pd.DataFrame:
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        return yf.download(ticker, period="5d", progress=False, timeout=10)


def resolve_symbol(raw: str) -> str:
    symbol = raw.strip().upper()
    if not symbol:
        return symbol
    if "." in symbol or symbol.startswith("^"):
        return symbol

    for suffix in (".NS", ".BO"):
        ticker = symbol + suffix
        try:
            probe = _quiet_yfinance_download(ticker)
            if probe is not None and not probe.empty:
                print(f"Resolved to {ticker}")
                return ticker
        except Exception:
            pass
    return symbol


def _fetch_one(raw_symbol: str, verbose: bool = True) -> tuple[str, Optional[pd.DataFrame]]:
    symbol = resolve_symbol(raw_symbol)
    df = get_stock_data(symbol, verbose=verbose)
    return symbol, df


def _fetch_many(symbols: list[str], verbose: bool = True) -> dict[str, pd.DataFrame]:
    if not symbols:
        return {}
    resolved = [resolve_symbol(symbol) for symbol in symbols]
    datasets = get_multiple_stocks(resolved, verbose=verbose)
    if datasets:
        return datasets

    fallback = {}
    for symbol in resolved:
        _, df = _fetch_one(symbol, verbose=verbose)
        if df is not None and not df.empty:
            fallback[symbol] = df
    return fallback


def _build_handler(symbol: str, df: pd.DataFrame):
    from fundamentals.financials import FinancialDataHandler, OHLCV

    handler = FinancialDataHandler()
    candles = [
        OHLCV(
            timestamp=index.to_pydatetime(),
            open=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(row["Close"]),
            volume=int(row["Volume"]),
        )
        for index, row in df.iterrows()
    ]
    handler.add_ohlcv(symbol, candles)
    return handler


def _infer_fundamental_inputs(symbol: str, df: pd.DataFrame) -> dict:
    close = float(df["Close"].iloc[-1])
    returns = df["Close"].pct_change().dropna()
    ret_20 = float(df["Close"].pct_change(20).iloc[-1] * 100) if len(df) > 21 else 5.0
    ret_60 = float(df["Close"].pct_change(60).iloc[-1] * 100) if len(df) > 61 else ret_20 * 1.5
    annual_vol = float(returns.rolling(20).std().iloc[-1] * np.sqrt(252) * 100) if len(returns) > 20 else 25.0

    market_cap = max(close * 1_000_000, 1_000_000.0)
    revenue = market_cap / max(2.1 + annual_vol / 40.0, 0.7)
    fcf = revenue * max(0.08 + ret_20 / 500.0, 0.02)
    equity = market_cap / max(1.5 + annual_vol / 80.0, 0.5)
    debt = equity * max(0.45 - ret_60 / 300.0, 0.1)
    shares_outstanding = max(market_cap / max(close, 1.0), 1.0)
    net_income = max(close * shares_outstanding / max(12.0 + annual_vol / 5.0, 5.0), 1.0)
    ebitda = revenue * 0.2

    return {
        "symbol": symbol,
        "current_price": close,
        "market_cap": market_cap,
        "revenue": revenue,
        "gross_profit": revenue * 0.48,
        "operating_income": revenue * 0.18,
        "net_income": net_income,
        "ebitda": ebitda,
        "assets": equity + debt + market_cap * 0.1,
        "liabilities": max((equity + debt) * 0.8, debt),
        "equity": equity,
        "operating_cash_flow": fcf * 1.15,
        "free_cash_flow": fcf,
        "debt": debt,
        "cash": max(market_cap * 0.05, 1.0),
        "pe": max(close / max(net_income / shares_outstanding, 1e-9), 1.0),
        "pb": max(market_cap / max(equity, 1.0), 0.5),
        "ps": max(market_cap / max(revenue, 1.0), 0.5),
        "ev_ebitda": max((market_cap + debt) / max(ebitda, 1.0), 3.0),
        "roe": (net_income / max(equity, 1.0)) * 100,
        "debt_to_equity": debt / max(equity, 1.0),
        "current_ratio": (equity + debt + market_cap * 0.1) / max((equity + debt) * 0.8, 1.0),
        "revenue_growth": max(ret_60 / 4.0, -10.0),
        "earnings_growth": max(ret_20 / 2.0, -10.0),
        "fcf_growth": max((ret_20 + ret_60) / 5.0, -10.0),
        "industry_pe": 20.0,
        "industry_pb": 3.0,
        "industry_ps": 2.0,
        "shares_outstanding": shares_outstanding,
        "net_debt": max(debt - market_cap * 0.05, 0.0),
        "eps": net_income / shares_outstanding,
        "bvps": equity / shares_outstanding,
        "sales_ps": revenue / shares_outstanding,
        "quality_score": float(np.clip(50 + ret_60 / 2.0 - annual_vol / 4.0, 5, 95)),
    }


def _select_symbols(multiple: bool = True) -> list[str]:
    _section("Universe Selection")
    print("1. Custom symbols")
    print("2. Default India watchlist")
    print("3. Default US watchlist")
    print("4. Nifty 50")
    print("5. Nifty 500")
    print("6. NSE sector")
    print("7. NSE search")
    choice = _ask("Choose universe", "1")

    if choice == "2":
        symbols = STOCKS
    elif choice == "3":
        symbols = US_STOCKS
    elif choice == "4":
        symbols = get_nifty50(use_fallback=True)
    elif choice == "5":
        symbols = get_nifty500(use_fallback=True)
    elif choice == "6":
        sector = _ask("Sector (it, bank, pharma, auto, nifty50, nifty500)", "it")
        symbols = get_sector(sector)
    elif choice == "7":
        query = _ask("Search text", "reliance")
        matches = search_symbol(query)
        if not matches:
            print("No matches found.")
            return []
        for idx, match in enumerate(matches, start=1):
            print(f"{idx}. {match['symbol']:<18} {match['name']}")
        pick = max(1, min(_ask_int("Select result", 1), len(matches)))
        symbols = [matches[pick - 1]["symbol"]]
    else:
        raw = _ask("Symbols separated by commas", "RELIANCE, TCS, INFY" if multiple else "RELIANCE")
        symbols = [part.strip() for part in raw.split(",") if part.strip()]
        if symbols and all("." not in item and not item.startswith("^") for item in symbols):
            symbols = get_custom_watchlist(symbols)

    resolved = [resolve_symbol(symbol) for symbol in symbols]
    unique = []
    seen = set()
    for symbol in resolved:
        if symbol and symbol not in seen:
            unique.append(symbol)
            seen.add(symbol)
    return unique if multiple else _choose_from_symbols(unique, "Symbol Selection")


def _load_random_forest_signal():
    from ai.random_forest_signal import rf_signal

    return rf_signal


def _load_lstm_predictor():
    from ai.lstm_predictor import lstm_predict

    return lstm_predict


def _load_trading_agent():
    from ai.reinforcement_agent import TradingAgent

    return TradingAgent


def _show_quote(symbol: str) -> None:
    quote = _safe(get_latest_price, symbol)
    if isinstance(quote, dict) and "error" not in quote:
        _subsection("Latest Quote")
        _print_mapping(
            {
                "Price": fmt_price(quote["price"]),
                "Change": f"{quote['change']:+.2f} ({quote['change_pct']:+.2f}%)",
                "Volume": fmt_volume(quote["volume"]),
                "Day Range": f"{fmt_price(quote['low'])} to {fmt_price(quote['high'])}",
                "Date": quote["date"],
                "Trend": quote["trend"],
            }
        )
        quote_score = _average_score(
            _pct_score(float(quote["change_pct"]), scale=2.0),
            _signal_score(quote["trend"]),
        )
        _print_verdict("Quote", quote_score)


def _print_mini_block(title: str, rows: list[tuple[str, str]]) -> None:
    from core.reporting import render_ascii_table

    print(render_ascii_table(title, rows))


def run_deep_analysis() -> None:
    from ai.premium_scoring import generate_premium_scores
    from ai.price_prediction import predict_price
    from ai.ranking_engine import detailed_score_report, stock_score
    from ai.signal_model import ai_signal
    from core.market_data_engine import fetch_market_snapshot
    from core.risk_management import build_risk_plan
    from fundamentals.analysis_engine import build_fundamental_report
    from fundamentals.ratios import TechnicalIndicators
    from fundamentals.valuation import AnomalyDetector
    from intelligence.premium_news import news_sentiment_snapshot
    from intelligence.block_deals import detect_block_deal
    from intelligence.insider_trading import insider_signal
    from intelligence.promoter_pledge import pledge_risk
    from options.options_strategy_builder import bull_call_spread
    from scanners.value_scanner import ValueScanner
    from technical.advanced_analysis import advanced_technical_report
    from technical.breakout import breakout_analysis
    from technical.elliott_wave import detect_elliott_wave
    from technical.fibonacci import fibonacci_retracements
    from technical.ichimoku import ichimoku_signal
    from technical.momentum import momentum_report
    from technical.moving_averages import ma_signal
    from technical.supertrend import supertrend_signal
    from technical.support_resistance import get_sr_zones, get_support_resistance
    from technical.trend_detection import detect_trend, trend_analysis
    from technical.volatility import volatility_report
    from technical.volume_profile import volume_profile, volume_profile_full
    from technical.vwap import anchored_vwap, vwap_bands

    symbols = _select_symbols(multiple=False)
    if not symbols:
        return
    symbol, df = _fetch_one(symbols[0], verbose=True)
    if df is None or df.empty:
        return

    market_snapshot = fetch_market_snapshot(symbol)
    lower_df = _lowercase_ohlcv(df)
    handler = _build_handler(symbol, df)
    momentum_20d = handler.get_price_momentum(symbol).get("20d", 0)
    _section(f"Deep Analysis - {symbol}")
    _print_mapping(
        {
            "Current Price": fmt_price(float(df["Close"].iloc[-1])),
            "Bars": len(df),
            "Date Range": f"{df.index[0].date()} to {df.index[-1].date()}",
            "Status": "Running full analysis pipeline...",
        }
    )

    _progress("Building technical layer")
    trend = detect_trend(df)
    trend_detail = _safe(trend_analysis, df)
    breakout = _safe(breakout_analysis, df)
    support_resistance = _safe(get_support_resistance, df)
    sr_zones = _safe(get_sr_zones, df)
    volume_poc = _safe(volume_profile, df)
    volume_profile_result = _safe(volume_profile_full, df)
    momentum = _safe(momentum_report, df)
    moving_averages = _safe(ma_signal, df)
    supertrend = _safe(supertrend_signal, df)
    volatility = _safe(volatility_report, df)
    vwap = _safe(vwap_bands, df)
    anchored_vwap_tail = _safe(lambda x: anchored_vwap(x).tail(3).round(2).to_list(), df)
    fibonacci = _safe(fibonacci_retracements, df)
    ichimoku = _safe(ichimoku_signal, df)
    elliott_wave = _safe(detect_elliott_wave, df)
    technical_score = _average_score(
        _signal_score(trend),
        _signal_score(breakout),
        _signal_score(moving_averages),
        _signal_score(supertrend),
        _signal_score(ichimoku),
        _pct_score(momentum_20d),
    )
    _progress("Technical layer complete")

    _progress("Running core AI models")
    prediction = _safe(predict_price, df, 5)
    ai_signal_result = _safe(ai_signal, df)
    ai_score_result = _safe(stock_score, df)
    ai_score_detail = _safe(detailed_score_report, df)

    try:
        _progress("Training random forest model")
        rf_signal = _load_random_forest_signal()
        rf_result = _safe(rf_signal, df)
    except Exception as exc:
        rf_result = f"unavailable ({exc})"

    try:
        _progress("Training LSTM predictor")
        lstm_predict = _load_lstm_predictor()
        lstm_result = _safe(lstm_predict, df, days_ahead=3)
    except Exception as exc:
        lstm_result = f"unavailable ({exc})"

    try:
        _progress("Training reinforcement learning agent")
        TradingAgent = _load_trading_agent()
        agent = TradingAgent()
        rl_result = _safe(agent.train, df, episodes=20)
    except Exception as exc:
        rl_result = f"unavailable ({exc})"
    ai_layer_score = _average_score(
        ai_signal_result.get("score") if isinstance(ai_signal_result, dict) else None,
        float(ai_score_result) if isinstance(ai_score_result, (int, float, np.floating)) else None,
        _prediction_score(prediction),
        _signal_score(rf_result, default=55.0),
        _signal_score(lstm_result, default=55.0),
        _signal_score(rl_result, default=50.0),
    )
    _progress("AI layer complete")

    _progress("Building fundamentals layer")
    inferred = _infer_fundamental_inputs(symbol, df)
    indicators = TechnicalIndicators().calculate_all_indicators(lower_df)
    fundamentals_score = _average_score(
        inferred["quality_score"],
        _rsi_score_value(float(indicators["rsi_14"][-1])),
        _pct_score(float(inferred["revenue_growth"])),
        _pct_score(float(inferred["earnings_growth"])),
    )
    _progress("Fundamentals layer complete")

    _progress("Running value and intelligence checks")
    value_result = _safe(
        ValueScanner().scan_value_stock,
        symbol=symbol,
        current_price=inferred["current_price"],
        pe=inferred["pe"],
        pb=inferred["pb"],
        ps=inferred["ps"],
        ev_ebitda=inferred["ev_ebitda"],
        market_cap=inferred["market_cap"],
        fcf=inferred["free_cash_flow"],
        roe=inferred["roe"],
        debt_to_equity=inferred["debt_to_equity"],
        current_ratio=inferred["current_ratio"],
        revenue_growth=inferred["revenue_growth"],
        earnings_growth=inferred["earnings_growth"],
        fcf_growth=inferred["fcf_growth"],
    )
    block_deal_signal = detect_block_deal(float(df['Volume'].iloc[-1]), float(df['Volume'].tail(20).mean()))
    insider = insider_signal(120, 45)
    pledge = pledge_risk(18)
    value_intelligence_score = _average_score(
        getattr(value_result, "score", None) if not isinstance(value_result, dict) else value_result.get("score"),
        _extract_confidence(value_result),
        _signal_score(getattr(value_result, "grade", None) if not isinstance(value_result, dict) else value_result.get("grade")),
        _signal_score(block_deal_signal),
        _signal_score(insider),
        _signal_score(pledge),
        _average_score(_pct_score(momentum_20d), 60.0),
    )
    _progress("Formatting final report")
    _print_deep_analysis_report(
        symbol=symbol,
        df=df,
        handler=handler,
        inferred=inferred,
        technical_score=technical_score,
        ai_layer_score=ai_layer_score,
        fundamentals_score=fundamentals_score,
        composite_score=value_intelligence_score,
        trend=trend,
        trend_detail=trend_detail,
        breakout=breakout,
        support_resistance=support_resistance,
        sr_zones=sr_zones,
        volume_poc=volume_poc,
        volume_profile_result=volume_profile_result,
        momentum=momentum,
        moving_averages=moving_averages,
        supertrend=supertrend,
        volatility=volatility,
        vwap=vwap,
        anchored_vwap_tail=anchored_vwap_tail,
        fibonacci=fibonacci,
        ichimoku=ichimoku,
        elliott_wave=elliott_wave,
        prediction=prediction,
        ai_signal_result=ai_signal_result,
        ai_score_result=ai_score_result,
        ai_score_detail=ai_score_detail,
        rf_result=rf_result,
        lstm_result=lstm_result,
        rl_result=rl_result,
        value_result=value_result,
        block_deal_signal=block_deal_signal,
        insider=insider,
        pledge=pledge,
    )

    premium_technical = advanced_technical_report(df)
    premium_fundamental = build_fundamental_report(inferred)
    premium_scores = generate_premium_scores(premium_technical, premium_fundamental, ai_signal_result, prediction)
    risk_plan = build_risk_plan(
        current_price=float(df["Close"].iloc[-1]),
        support=float(premium_technical.get("support", 0.0)),
        resistance=float(premium_technical.get("resistance", 0.0)),
        volatility_regime=str(premium_technical.get("volatility_regime", "Normal")),
    )
    premium_news = news_sentiment_snapshot(symbol)

    _subsection("Premium Intelligence Layer")
    _print_mini_block(
        "Smart Market Data Engine",
        [
            ("Resolved Symbol", market_snapshot.symbol),
            ("History Source", str(market_snapshot.metadata.get("history_source"))),
            ("Actions Source", str(market_snapshot.metadata.get("actions_source"))),
            ("Live Price", fmt_price(float(market_snapshot.live_price.get("price", float(df["Close"].iloc[-1]))))),
            ("Corporate Actions", str(len(market_snapshot.corporate_actions))),
        ],
    )
    _print_mini_block(
        "Advanced Technical Analysis",
        [
            ("Trend", str(premium_technical.get("trend"))),
            ("Trend Strength", str(premium_technical.get("trend_strength"))),
            ("Momentum Score", f"{premium_technical.get('momentum_score')}/100"),
            ("Volatility Regime", str(premium_technical.get("volatility_regime"))),
            ("Confluence", f"{premium_technical.get('confluence_score')}/100"),
        ],
    )
    _print_mini_block(
        "Fundamental Analysis Engine",
        [
            ("PE / PB", f"{premium_fundamental.get('pe')} / {premium_fundamental.get('pb')}"),
            ("ROE / ROCE", f"{premium_fundamental.get('roe')}% / {premium_fundamental.get('roce')}%"),
            ("Revenue Growth", f"{premium_fundamental.get('revenue_growth')}%"),
            ("Debt Health", str(premium_fundamental.get("debt_health"))),
            ("Fair Value", fmt_price(float(premium_fundamental.get("fair_value", 0.0)))),
        ],
    )
    _print_mini_block(
        "AI Ranking and Risk",
        [
            ("Action", str(premium_scores.get("action"))),
            ("Buy Score", f"{premium_scores.get('buy_score')}/100"),
            ("Swing Score", f"{premium_scores.get('swing_score')}/100"),
            ("Long-Term Score", f"{premium_scores.get('long_term_score')}/100"),
            ("Risk", f"{premium_scores.get('risk_label')} ({premium_scores.get('risk_score')}/100)"),
            ("Confidence", f"{premium_scores.get('confidence_label')} ({premium_scores.get('confidence')}/100)"),
        ],
    )
    _print_mini_block(
        "Risk Management Module",
        [
            ("Stop Loss", fmt_price(float(risk_plan.get("stop_loss", 0.0)))),
            ("Target Price", fmt_price(float(risk_plan.get("target_price", 0.0)))),
            ("Position Size", str(risk_plan.get("position_size_shares"))),
            ("Capital at Risk", fmt_price(float(risk_plan.get("capital_at_risk", 0.0)))),
            ("Risk / Reward", str(risk_plan.get("risk_reward_ratio"))),
            ("Drawdown Warning", str(risk_plan.get("max_drawdown_warning"))),
        ],
    )
    print(f"Latest News Sentiment    {premium_news.get('overall_sentiment')} ({premium_news.get('average_score')})")
    if premium_news.get("event_risk_alerts"):
        print(f"Event Risk Alerts        {premium_news.get('event_risk_alerts')[:3]}")
    print(f"Earnings Alert           {premium_news.get('earnings_alert')}")


def run_scanner_suite() -> None:
    from scanners.breakout_scanner import BreakoutScanner
    from scanners.momentum_scanner import MomentumScanner
    from scanners.premium_scanners import run_premium_scanners
    from scanners.swing_scanner import SwingScanner
    from scanners.value_scanner import ValueScanner
    from fundamentals.analysis_engine import build_fundamental_report

    symbols = _select_symbols(multiple=True)
    datasets = _fetch_many(symbols, verbose=True)
    if not datasets:
        return
    lower_datasets = {symbol: _lowercase_ohlcv(df) for symbol, df in datasets.items()}

    _section("Scanner Suite")
    print("Breakout Scanner")
    breakout_scanner = BreakoutScanner()
    breakout_results = breakout_scanner.scan_multiple(lower_datasets)
    if breakout_results:
        for result in breakout_results[:5]:
            print(breakout_scanner.format_setup(result))
        breakout_score = _average_score(
            breakout_results[0].confidence,
            breakout_results[0].breakout_strength,
            _signal_score(breakout_results[0].signal),
        )
    else:
        print("No breakout setups found.")
        breakout_score = 45.0
    _print_verdict("Breakout", breakout_score)

    print("\nMomentum Scanner")
    momentum_scanner = MomentumScanner()
    momentum_results = momentum_scanner.scan_stocks(lower_datasets)
    if momentum_results:
        for result in momentum_results[:5]:
            print(momentum_scanner.format_analysis(result))
        momentum_score = _average_score(
            momentum_results[0].momentum_score,
            momentum_results[0].trend_strength,
            float(momentum_results[0].rating) * 20,
            _signal_score(momentum_results[0].signal),
        )
    else:
        print("No momentum setups found.")
        momentum_score = 45.0
    _print_verdict("Momentum", momentum_score)

    print("\nSwing Scanner")
    swing_scanner = SwingScanner()
    swing_results = swing_scanner.scan_multiple(lower_datasets)
    if swing_results:
        for result in swing_results[:5]:
            print(swing_scanner.format_setup(result))
        swing_score = _average_score(
            swing_results[0].confidence,
            swing_results[0].technical_score,
            swing_results[0].strength * 100,
            _signal_score(swing_results[0].signal),
        )
    else:
        print("No swing setups found.")
        swing_score = 45.0
    _print_verdict("Swing", swing_score)

    print("\nValue Scanner")
    value_inputs = {}
    for symbol, df in datasets.items():
        inferred = _infer_fundamental_inputs(symbol, df)
        value_inputs[symbol] = {
            "current_price": inferred["current_price"],
            "pe": inferred["pe"],
            "pb": inferred["pb"],
            "ps": inferred["ps"],
            "ev_ebitda": inferred["ev_ebitda"],
            "market_cap": inferred["market_cap"],
            "fcf": inferred["free_cash_flow"],
            "roe": inferred["roe"],
            "debt_to_equity": inferred["debt_to_equity"],
            "current_ratio": inferred["current_ratio"],
            "revenue_growth": inferred["revenue_growth"],
            "earnings_growth": inferred["earnings_growth"],
            "fcf_growth": inferred["fcf_growth"],
        }
    value_results = ValueScanner().scan_portfolio(value_inputs)
    if value_results:
        print("Value inputs are assumption-driven because there is no live statements pipeline wired in this repo.")
        for result in value_results[:5]:
            print(ValueScanner().format_analysis(result))
        value_score = _average_score(
            value_results[0].score,
            value_results[0].confidence,
            _signal_score(value_results[0].grade),
        )
    else:
        print("No value setups found.")
        value_score = 45.0
    _print_verdict("Value", value_score)

    premium_fundamentals = {symbol: build_fundamental_report(payload) for symbol, payload in value_inputs.items()}
    premium_scans = run_premium_scanners(datasets, premium_fundamentals)
    print("\nPremium Scanners")
    for label, items in premium_scans.items():
        print(f"{label.replace('_', ' ').title():<30} {len(items)} matches")
        for item in items[:3]:
            print(f"  {item}")
    _print_verdict("Scanner Suite", _average_score(breakout_score, momentum_score, swing_score, value_score))


def run_fundamentals_suite() -> None:
    from fundamentals.financials import FinancialMetrics
    from fundamentals.peer_comparison import (
        ClusteringAnalyzer,
        ComparisonMetric,
        PeerComparator,
        PeerMetrics,
        SectorAnalyzer,
    )
    from fundamentals.ratios import RatioAnalyzer, TechnicalIndicators
    from fundamentals.valuation import AnomalyDetector, DCFValuation, GrowthForecasting, ValuationEngine

    symbols = _select_symbols(multiple=False)
    if not symbols:
        return
    symbol, df = _fetch_one(symbols[0], verbose=True)
    if df is None or df.empty:
        return
    lower_df = _lowercase_ohlcv(df)
    inferred = _infer_fundamental_inputs(symbol, df)

    _section(f"Fundamentals and Valuation - {symbol}")

    _subsection("Financial Data Handler")
    handler = _build_handler(symbol, df)
    metrics = FinancialMetrics(
        revenue=inferred["revenue"],
        gross_profit=inferred["gross_profit"],
        operating_income=inferred["operating_income"],
        net_income=inferred["net_income"],
        ebitda=inferred["ebitda"],
        assets=inferred["assets"],
        liabilities=inferred["liabilities"],
        equity=inferred["equity"],
        operating_cash_flow=inferred["operating_cash_flow"],
        free_cash_flow=inferred["free_cash_flow"],
        debt=inferred["debt"],
        cash=inferred["cash"],
    )
    handler.add_financial_metrics(symbol, [metrics])
    _print_mapping(
        {
            "Latest Price": fmt_price(handler.get_latest_price(symbol) or 0),
            "52w Range": f"{fmt_price(handler.get_price_range(symbol, 252)[0])} to {fmt_price(handler.get_price_range(symbol, 252)[1])}",
            "Volatility": f"{handler.get_volatility(symbol):.2f}%",
            "Momentum 50d": f"{handler.get_price_momentum(symbol).get('50d', 0):.2f}%",
            "Gap Count": len(handler.detect_gaps(symbol)),
        }
    )
    handler_score = _average_score(
        inferred["quality_score"],
        _pct_score(handler.get_price_momentum(symbol).get("50d", 0)),
        _clip_score(100 - handler.get_volatility(symbol)),
    )
    _print_verdict("Handler", handler_score)

    _subsection("Technical Indicator Engine")
    indicators = TechnicalIndicators().calculate_all_indicators(lower_df)
    last_rsi = float(indicators["rsi_14"][-1])
    last_macd_hist = float(indicators["macd_histogram"][-1])
    print(
        f"RSI={last_rsi:.2f} | "
        f"MACD Hist={last_macd_hist:.4f} | "
        f"ATR={indicators['atr_14'][-1]:.2f} | "
        f"CCI={indicators['cci'][-1]:.2f}"
    )
    print(f"Multiples               {RatioAnalyzer.calculate_multiples(pd.DataFrame([{'price': inferred['current_price'], 'eps': inferred['eps'], 'pb': inferred['pb'], 'ps': inferred['ps']}]))}")
    indicator_score = _average_score(
        _rsi_score_value(last_rsi),
        _clip_score(50 + last_macd_hist * 250),
        _pct_score(inferred["earnings_growth"]),
    )
    _print_verdict("Indicators", indicator_score)

    _subsection("Valuation Engine")
    growth_rates = [
        inferred["revenue_growth"] / 100.0,
        inferred["earnings_growth"] / 100.0,
        inferred["fcf_growth"] / 100.0,
        0.04,
        0.03,
    ]
    engine = ValuationEngine()
    valuation_results = engine.comprehensive_valuation(
        current_price=inferred["current_price"],
        base_fcf=inferred["free_cash_flow"],
        growth_rates=growth_rates,
        shares_outstanding=inferred["shares_outstanding"],
        net_debt=inferred["net_debt"],
        comparable_multiples={
            "pe": [18, 20, 22],
            "pb": [2.5, 3.0, 3.5],
            "ps": [1.8, 2.0, 2.2],
        },
        eps=inferred["eps"],
        bvps=inferred["bvps"],
        sales_ps=inferred["sales_ps"],
    )
    for item in valuation_results:
        print(item.to_dict())
    sensitivity = DCFValuation().sensitivity_analysis(
        inferred["free_cash_flow"],
        growth_rates,
        inferred["shares_outstanding"],
        inferred["net_debt"],
        (0.07, 0.11),
        (0.02, 0.04),
    )
    print(f"Sensitivity Sample      {list(sensitivity.items())[:3]}")
    print(f"Exponential Forecast    {GrowthForecasting.exponential_smoothing(df['Close'].tail(12).to_list(), periods=3)}")
    print(f"ARIMA-like Forecast     {GrowthForecasting.arima_simple_forecast(df['Close'].tail(12).to_list(), periods=3)}")
    print(f"Volatility Regimes      {AnomalyDetector.volatility_regimes(df['Close'].pct_change().dropna().values)[-5:]}")
    valuation_score = _average_score(
        _average_score(*[_upside_score(item.upside_downside) for item in valuation_results]) if valuation_results else None,
        _average_score(*[item.confidence for item in valuation_results]) if valuation_results else None,
        _average_score(*[_signal_score(item.recommendation) for item in valuation_results]) if valuation_results else None,
    )
    _print_verdict("Valuation", valuation_score)

    _subsection("Peer Comparison")
    peer_symbols = _select_symbols(multiple=True)
    peer_data = _fetch_many(([symbol] + peer_symbols)[:6], verbose=False)
    comparator = PeerComparator()
    sector_analyzer = SectorAnalyzer()
    peer_objects = []
    sector_name = _ask("Sector label", "General")
    industry_name = _ask("Industry label", "General")

    for peer_symbol, peer_df in peer_data.items():
        peer_inputs = _infer_fundamental_inputs(peer_symbol, peer_df)
        peer_metrics = PeerMetrics(
            ticker=peer_symbol,
            company_name=peer_symbol,
            sector=sector_name,
            industry=industry_name,
            market_cap=peer_inputs["market_cap"],
            metrics={
                ComparisonMetric.PE_RATIO: peer_inputs["pe"],
                ComparisonMetric.PB_RATIO: peer_inputs["pb"],
                ComparisonMetric.PS_RATIO: peer_inputs["ps"],
                ComparisonMetric.ROE: peer_inputs["roe"],
                ComparisonMetric.ROA: peer_inputs["roe"] / 2.0,
                ComparisonMetric.DEBT_TO_EQUITY: peer_inputs["debt_to_equity"],
                ComparisonMetric.CURRENT_RATIO: peer_inputs["current_ratio"],
                ComparisonMetric.FCF_MARGIN: peer_inputs["free_cash_flow"] / max(peer_inputs["revenue"], 1.0) * 100,
                ComparisonMetric.REVENUE_GROWTH: peer_inputs["revenue_growth"],
                ComparisonMetric.EPS_GROWTH: peer_inputs["earnings_growth"],
                ComparisonMetric.DIVIDEND_YIELD: 2.0,
                ComparisonMetric.PRICE_MOMENTUM: float(peer_df["Close"].pct_change(60).iloc[-1] * 100) if len(peer_df) > 61 else 0.0,
            },
            growth_rate=peer_inputs["revenue_growth"],
            quality_score=peer_inputs["quality_score"],
        )
        comparator.add_peer(peer_metrics)
        peer_objects.append(peer_metrics)

    sector_analyzer.add_sector_data(sector_name, peer_objects)
    comparison = comparator.comprehensive_comparison(symbol)
    print(comparison)
    print(f"Sector Stats             {sector_analyzer.calculate_sector_statistics(sector_name)}")
    sector_health = sector_analyzer.sector_health_score(sector_name)
    print(f"Sector Health            {sector_health:.2f}")
    print(f"Clusters                 {ClusteringAnalyzer.identify_clusters(peer_objects, n_clusters=min(3, len(peer_objects)))}")
    print("Peer analytics here are scenario tools because the repo lacks a real fundamentals ingestion source.")
    peer_score = _average_score(
        sector_health,
        _average_score(*[peer.quality_score for peer in peer_objects]) if peer_objects else None,
        _average_score(*[_pct_score(peer.growth_rate) for peer in peer_objects]) if peer_objects else None,
    )
    _print_verdict("Peers", peer_score)
    _print_verdict("Fundamentals", _average_score(handler_score, indicator_score, valuation_score, peer_score))


def run_ai_lab() -> None:
    from ai.price_prediction import predict_price
    from ai.ranking_engine import stock_score
    from ai.sentiment_model import analyze_batch, analyze_news
    from ai.signal_model import ai_signal

    print()
    print("1. AI signal")
    print("2. Price prediction")
    print("3. Random forest")
    print("4. LSTM predictor")
    print("5. Reinforcement learning agent")
    print("6. Sentiment")
    print("7. Full AI suite")
    choice = _ask("Choose AI feature", "7")

    if choice == "6":
        if _ask_yes_no("Use batch headlines", False):
            raw = _ask("Headlines separated by |", "Strong earnings growth | Broker upgrades target")
            headlines = [item.strip() for item in raw.split("|") if item.strip()]
            _section("Sentiment Batch")
            sentiment_batch = analyze_batch(headlines)
            print(sentiment_batch)
            _print_verdict("Sentiment", _upside_score(float(sentiment_batch.get("average_score", 0)) * 100))
        else:
            headline = _ask("Headline text", "Company posts record quarterly growth and margin expansion")
            _section("Sentiment")
            sentiment = analyze_news(headline)
            print(sentiment)
            _print_verdict("Sentiment", _upside_score(float(sentiment.get("score", 0)) * 100))
        return

    symbols = _select_symbols(multiple=False)
    if not symbols:
        return
    symbol, df = _fetch_one(symbols[0], verbose=True)
    if df is None or df.empty:
        return

    _section(f"AI Lab - {symbol}")
    ai_scores = []
    if choice in {"1", "7"}:
        ai_signal_result = _safe(ai_signal, df)
        print(f"AI Signal                {ai_signal_result}")
        if isinstance(ai_signal_result, dict) and "score" in ai_signal_result:
            ai_scores.append(float(ai_signal_result["score"]))
    if choice in {"2", "7"}:
        prediction = _safe(predict_price, df, 5)
        print(f"Price Prediction         {prediction}")
        prediction_score = _prediction_score(prediction)
        if prediction_score is not None:
            ai_scores.append(prediction_score)
    if choice in {"3", "7"}:
        try:
            rf_signal = _load_random_forest_signal()
            rf_result = _safe(rf_signal, df)
            print(f"Random Forest            {rf_result}")
            ai_scores.append(_signal_score(rf_result, default=55.0))
        except Exception as exc:
            print(f"Random Forest            unavailable ({exc})")
    if choice in {"4", "7"}:
        try:
            lstm_predict = _load_lstm_predictor()
            lstm_result = _safe(lstm_predict, df, days_ahead=5)
            print(f"LSTM Predictor           {lstm_result}")
            ai_scores.append(_signal_score(lstm_result, default=55.0))
        except Exception as exc:
            print(f"LSTM Predictor           unavailable ({exc})")
    if choice in {"5", "7"}:
        try:
            TradingAgent = _load_trading_agent()
            agent = TradingAgent()
            rl_result = _safe(agent.train, df, episodes=_ask_int('Episodes', 30))
            print(f"RL Train Summary         {rl_result}")
            ai_scores.append(_signal_score(rl_result, default=50.0))
        except Exception as exc:
            print(f"RL Agent                 unavailable ({exc})")
    ai_scores.append(float(stock_score(df)))
    _print_verdict("AI Lab", _average_score(*ai_scores))


def run_portfolio_analytics() -> None:
    from portfolio.optimizer import equal_weight_portfolio, optimize_portfolio
    from portfolio.premium_optimizer import portfolio_health_report, sip_simulation

    symbols = _select_symbols(multiple=True)
    if len(symbols) < 2:
        print("Need at least two symbols for portfolio analytics.")
        return
    datasets = _fetch_many(symbols, verbose=True)
    if len(datasets) < 2:
        print("Could not fetch enough symbols.")
        return

    _section("Portfolio Analytics")
    print(f"Equal Weight             {equal_weight_portfolio(list(datasets.keys()))}")
    optimize_equal = optimize_portfolio(datasets, method='equal_weight')
    optimize_max_sharpe = optimize_portfolio(datasets, method='max_sharpe', n_simulations=1500)
    optimize_min_vol = optimize_portfolio(datasets, method='min_volatility', n_simulations=1500)
    print(f"Optimize Equal           {optimize_equal}")
    print(f"Optimize Max Sharpe      {optimize_max_sharpe}")
    print(f"Optimize Min Vol         {optimize_min_vol}")

    aligned = pd.DataFrame({symbol: df['Close'] for symbol, df in datasets.items()}).dropna()
    returns = aligned.pct_change().dropna()
    if not returns.empty:
        equal_returns = returns.mean(axis=1).values
        sharpe = round(sharpe_ratio(equal_returns), 4)
        sortino = round(sortino_ratio(equal_returns), 4)
        cumulative_return = ((1 + pd.Series(equal_returns)).prod() - 1) * 100
        _print_mapping(
            {
                "Sharpe": sharpe,
                "Sortino": sortino,
                "Cumulative Return": f"{cumulative_return:.2f}%",
            }
        )
        portfolio_score = _average_score(
            _pct_score(float(cumulative_return)),
            _clip_score(50 + float(sharpe) * 15),
            _clip_score(50 + float(sortino) * 12),
            _clip_score(50 + float(optimize_max_sharpe.get("sharpe_ratio", 0)) * 15) if isinstance(optimize_max_sharpe, dict) else None,
        )
        _print_verdict("Portfolio", portfolio_score)
        premium_portfolio = portfolio_health_report(datasets, optimize_max_sharpe.get("weights", {}) if isinstance(optimize_max_sharpe, dict) else {})
        print(f"Premium Portfolio Health {premium_portfolio}")
        sip = sip_simulation(
            monthly_investment=_ask_float("Monthly SIP amount", 10000.0),
            annual_return_pct=_ask_float("Expected annual return %", 12.0),
            years=_ask_int("SIP years", 10),
        )
        print(f"SIP Simulation           {sip}")


def run_backtest() -> None:
    from ai.signal_model import ai_signal
    from core.advanced_backtester import compare_strategies
    from core.reporting import mini_chart
    from technical.breakout import breakout_signal
    from technical.ichimoku import ichimoku_signal
    from technical.moving_averages import ma_signal
    from technical.supertrend import supertrend_signal

    symbols = _select_symbols(multiple=False)
    if not symbols:
        return
    symbol, df = _fetch_one(symbols[0], verbose=True)
    if df is None or df.empty:
        return

    def ai_strategy(slice_df: pd.DataFrame) -> str:
        result = ai_signal(slice_df)
        return result.get("signal", "HOLD") if isinstance(result, dict) else "HOLD"

    def ma_strategy(slice_df: pd.DataFrame) -> str:
        result = ma_signal(slice_df)
        return result.get("signal", "HOLD") if isinstance(result, dict) else "HOLD"

    def breakout_strategy(slice_df: pd.DataFrame) -> str:
        return "BUY" if breakout_signal(slice_df) else "HOLD"

    def ichimoku_strategy(slice_df: pd.DataFrame) -> str:
        result = ichimoku_signal(slice_df)
        return result.get("signal", "HOLD") if isinstance(result, dict) else "HOLD"

    def supertrend_strategy(slice_df: pd.DataFrame) -> str:
        result = supertrend_signal(slice_df)
        direction = result.get("direction", "HOLD").upper() if isinstance(result, dict) else "HOLD"
        if "UP" in direction or "BUY" in direction:
            return "BUY"
        if "DOWN" in direction or "SELL" in direction:
            return "SELL"
        return "HOLD"

    strategies = {
        "1": ("AI Signal", ai_strategy),
        "2": ("Moving Average", ma_strategy),
        "3": ("Breakout", breakout_strategy),
        "4": ("Ichimoku", ichimoku_strategy),
        "5": ("Supertrend", supertrend_strategy),
    }

    print()
    for key, (name, _) in strategies.items():
        print(f"{key}. {name}")
    strategy_key = _ask("Choose strategy", "1")
    strategy_name, strategy_fn = strategies.get(strategy_key, strategies["1"])

    bt = Backtester(
        df=df,
        signal_fn=strategy_fn,
        initial_capital=_ask_float("Initial capital", 100000.0),
        position_size=_ask_float("Position size fraction", 0.1),
        stop_loss=_ask_float("Stop loss fraction", 0.05),
        take_profit=_ask_float("Take profit fraction", 0.15),
    )
    result = bt.run()

    _section(f"Backtest - {symbol} - {strategy_name}")
    Backtester.print_report(result)
    trades = Backtester.trade_log(result)
    if not trades.empty:
        print(trades.head(10).to_string(index=False))
    metrics = result.get("metrics", {})
    backtest_score = _average_score(
        _pct_score(float(metrics.get("total_return_pct", 0))),
        _clip_score(50 + float(metrics.get("sharpe_ratio", 0)) * 15),
        float(metrics.get("win_rate_pct", 50)),
        _clip_score(100 - abs(float(metrics.get("max_drawdown_pct", 0))) * 2),
    )
    _print_verdict("Backtest", backtest_score)
    comparison = compare_strategies(
        df=df,
        strategies={name: fn for name, fn in strategies.values()},
        initial_capital=bt.initial_capital,
        position_size=bt.position_size,
        stop_loss=bt.stop_loss,
        take_profit=bt.take_profit,
    )
    print(f"Strategy Comparison      {comparison}")
    print(f"Equity Curve             {mini_chart(result.get('equity_curve', []))}")


def run_market_intelligence() -> None:
    from ai.sentiment_model import analyze_news
    from intelligence.block_deals import detect_block_deal
    from intelligence.fii_dii import analyze_fii_dii
    from intelligence.insider_trading import insider_signal
    from intelligence.premium_news import news_sentiment_snapshot
    from intelligence.promoter_pledge import pledge_risk

    _section("Market Intelligence")
    headline = _ask("Headline", "Company beats estimates and raises guidance")
    news_sentiment = analyze_news(headline)
    print(f"News Sentiment           {news_sentiment}")

    insider_buy = _ask_float("Insider buy amount", 120.0)
    insider_sell = _ask_float("Insider sell amount", 45.0)
    insider = insider_signal(insider_buy, insider_sell)
    print(f"Insider Signal           {insider}")

    pledge = _ask_float("Promoter pledge %", 18.0)
    pledge_signal = pledge_risk(pledge)
    print(f"Pledge Risk              {pledge_signal}")

    if _ask_yes_no("Use live volume for block-deal check", True):
        symbols = _select_symbols(multiple=False)
        if symbols:
            symbol, df = _fetch_one(symbols[0], verbose=False)
            if df is not None and len(df) >= 20:
                current_volume = float(df["Volume"].iloc[-1])
                avg_volume = float(df["Volume"].tail(20).mean())
                block_deal = detect_block_deal(current_volume, avg_volume)
                print(f"Block Deal               {block_deal}")
            else:
                block_deal = "Normal"
        else:
            block_deal = "Normal"
    else:
        current_volume = _ask_float("Current volume", 3200000)
        avg_volume = _ask_float("Average volume", 900000)
        block_deal = detect_block_deal(current_volume, avg_volume)
        print(f"Block Deal               {block_deal}")

    fii_dii_df = pd.DataFrame(
        {
            "FII_Net": [120, 180, -50, 90, 110, 140],
            "DII_Net": [60, 40, 55, 35, 25, 20],
        }
    )
    fii_dii = analyze_fii_dii(fii_dii_df)
    print(f"FII DII Trend            {fii_dii}")
    if _ask_yes_no("Resolve a bare symbol", True):
        premium_symbol = resolve_symbol(_ask("Bare symbol", "RELIANCE"))
        premium_news = news_sentiment_snapshot(premium_symbol)
        print(f"Premium News Snapshot    {premium_news}")
    intelligence_score = _average_score(
        _upside_score(float(news_sentiment.get("score", 0)) * 100) if isinstance(news_sentiment, dict) else None,
        _signal_score(insider),
        _signal_score(pledge_signal),
        _signal_score(block_deal),
        _signal_score(fii_dii, default=55.0),
    )
    _print_verdict("Intelligence", intelligence_score)


def run_premium_command_center() -> None:
    from ai.premium_scoring import generate_premium_scores
    from core.advanced_backtester import compare_strategies
    from core.market_data_engine import fetch_market_snapshot
    from core.reporting import export_csv, export_text_pdf, mini_chart
    from core.risk_management import build_risk_plan
    from fundamentals.analysis_engine import build_fundamental_report
    from intelligence.premium_news import news_sentiment_snapshot
    from scanners.premium_scanners import run_premium_scanners
    from technical.advanced_analysis import advanced_technical_report

    symbols = _select_symbols(multiple=True)
    if not symbols:
        return
    datasets = _fetch_many(symbols, verbose=True)
    if not datasets:
        return

    _section("Premium Command Center")
    fundamentals_map = {}
    score_rows = []

    for symbol, df in datasets.items():
        inferred = _infer_fundamental_inputs(symbol, df)
        technical_report = advanced_technical_report(df)
        fundamental_report = build_fundamental_report(inferred)
        premium_scores = generate_premium_scores(technical_report, fundamental_report)
        risk_plan = build_risk_plan(
            current_price=float(df["Close"].iloc[-1]),
            support=float(technical_report.get("support", 0.0)),
            resistance=float(technical_report.get("resistance", 0.0)),
            volatility_regime=str(technical_report.get("volatility_regime", "Normal")),
        )
        market_snapshot = fetch_market_snapshot(symbol)
        news_snapshot = news_sentiment_snapshot(symbol)
        fundamentals_map[symbol] = fundamental_report
        score_rows.append([symbol, premium_scores["action"], premium_scores["buy_score"], premium_scores["risk_label"], premium_scores["confidence_label"]])

        print(f"\n{symbol}")
        print(f"Market Snapshot          source={market_snapshot.source} live={market_snapshot.live_price.get('price', 'N/A')}")
        print(f"Premium Scorecard        {premium_scores}")
        print(f"Risk Plan                {risk_plan}")
        print(f"News Snapshot            {news_snapshot.get('overall_sentiment')} | alerts={news_snapshot.get('event_risk_alerts')[:2]}")
        print(f"Price Chart              {mini_chart(df['Close'].tail(90).tolist())}")

    premium_scans = run_premium_scanners(datasets, fundamentals_map)
    print(f"\nPremium Scanner Snapshot {premium_scans}")

    first_symbol = next(iter(datasets))
    first_df = datasets[first_symbol]
    comparison = compare_strategies(
        df=first_df,
        strategies={
            "Trend Hold": lambda slice_df: "BUY" if slice_df["Close"].iloc[-1] > slice_df["Close"].rolling(20).mean().iloc[-1] else "SELL",
            "Breakout Hold": lambda slice_df: "BUY" if slice_df["Close"].iloc[-1] >= slice_df["High"].rolling(20).max().iloc[-1] else "HOLD",
        },
        initial_capital=100000.0,
        position_size=0.1,
        stop_loss=0.05,
        take_profit=0.15,
    )
    print(f"Strategy Compare         {comparison}")

    if _ask_yes_no("Export premium report", True):
        base_name = _ask("Export base filename", "premium_command_center")
        csv_path = export_csv(
            os.path.join(os.getcwd(), "exports", f"{base_name}.csv"),
            ["Symbol", "Action", "Buy Score", "Risk", "Confidence"],
            score_rows,
        )
        pdf_lines = [f"{row[0]} | {row[1]} | Buy {row[2]} | Risk {row[3]} | Confidence {row[4]}" for row in score_rows]
        pdf_path = export_text_pdf(
            os.path.join(os.getcwd(), "exports", f"{base_name}.pdf"),
            "Ultimate Stock AI Premium Report",
            pdf_lines,
        )
        print(f"Exported CSV             {csv_path}")
        print(f"Exported PDF             {pdf_path}")


def run_options_tools() -> None:
    from options.options_strategy_builder import bull_call_spread

    symbols = _select_symbols(multiple=False)
    underlying = None
    if symbols:
        _, df = _fetch_one(symbols[0], verbose=False)
        if df is not None and not df.empty:
            underlying = float(df["Close"].iloc[-1])
    if underlying is None:
        underlying = _ask_float("Underlying price", 100.0)

    _section("Options Tools")
    spread = bull_call_spread(underlying)
    _print_mapping(
        {
            "Underlying": fmt_price(underlying),
            "Buy Call Strike": fmt_price(spread["buy_call"]),
            "Sell Call Strike": fmt_price(spread["sell_call"]),
            "Spread Width": fmt_price(spread["sell_call"] - spread["buy_call"]),
        }
    )
    options_score = _average_score(
        60.0,
        _pct_score(((spread["sell_call"] / max(underlying, 1e-9)) - 1) * 100, scale=12),
    )
    _print_verdict("Options", options_score)


def run_symbol_tools() -> None:
    _section("Watchlists and Symbol Tools")
    print("Default India watchlist")
    _print_list(STOCKS)
    print("\nDefault US watchlist")
    _print_list(US_STOCKS)
    print("\nConfigured Nifty sample")
    _print_list(NIFTY50_STOCKS, limit=15)

    if _ask_yes_no("Run NSE search", False):
        query = _ask("Search text", "infosys")
        print(search_symbol(query))

    if _ask_yes_no("Resolve a bare symbol", True):
        raw = _ask("Bare symbol", "ITC")
        print(f"Resolved Symbol          {resolve_symbol(raw)}")


def run_repo_review() -> None:
    _section("Repo Review")
    print("Production-ready modules")
    _print_list(PRODUCTION_READY)
    print("\nBeta / usable with assumptions")
    _print_list(BETA_READY)
    print("\nExperimental or dependency-sensitive")
    _print_list(EXPERIMENTAL)
    print(
        "\nAssessment:\n"
        "- The market-data and core-technical pipeline is the strongest part of the repo.\n"
        "- Fundamentals, peer comparison, and value workflows are reusable engines but need real statements data to be decision-grade.\n"
        "- ML extras are promising but should be treated as optional research tools unless you validate dependencies, training, and outputs in production."
    )


def main() -> None:
    _setup_console()
    while True:
        print(MENU)
        choice = _ask("Choose feature", "1")
        try:
            if choice == "1":
                run_deep_analysis()
            elif choice == "2":
                run_scanner_suite()
            elif choice == "3":
                run_fundamentals_suite()
            elif choice == "4":
                run_ai_lab()
            elif choice == "5":
                run_portfolio_analytics()
            elif choice == "6":
                run_backtest()
            elif choice == "7":
                run_market_intelligence()
            elif choice == "8":
                run_options_tools()
            elif choice == "9":
                run_symbol_tools()
            elif choice == "10":
                run_repo_review()
            elif choice == "11":
                run_premium_command_center()
            elif choice == "0":
                print("Exiting terminal.")
                return
            else:
                print("Unknown choice.")
        except KeyboardInterrupt:
            print("\nAction cancelled.")
        except Exception as exc:
            print(f"\nUnexpected error: {exc}")
        _pause()


if __name__ == "__main__":
    main()
