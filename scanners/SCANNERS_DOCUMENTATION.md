╔════════════════════════════════════════════════════════════════════════════╗
║         ADVANCED TRADING SCANNERS - COMPLETE DOCUMENTATION v2.1           ║
║              Multi-Strategy Stock Screening & Analysis System             ║
╚════════════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Four specialized trading scanners that work independently or unified:

1. SWING SCANNER     - Short-term trading patterns & signals
2. VALUE SCANNER    - Fundamental value investing analysis  
3. MOMENTUM SCANNER - Trend and momentum-based signals
4. BREAKOUT SCANNER - Support/resistance and pattern breakouts

Plus:
5. UNIFIED SCANNER  - Combines all 4 scanners for consensus signals

════════════════════════════════════════════════════════════════════════════════

1️⃣ SWING SCANNER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PURPOSE: Identify short-term swing trading opportunities (3-10 day holds)

KEY FEATURES:
  ✓ Pattern recognition (double bottoms, ascending triangles, head & shoulders)
  ✓ Multi-indicator confirmation (RSI, MACD, volume)
  ✓ Support/resistance level detection
  ✓ Risk/reward ratio calculation
  ✓ Entry/stop/target levels
  ✓ Volume profile analysis
  ✓ Volatility (ATR) adjusted stops

SIGNALS:
  • "Strong Swing Buy" (confidence >75%, 4+ signals)
  • "Swing Buy" (confidence >50%, 3+ signals)
  • "Weak Swing Buy" (confidence >40%, 2+ signals)
  • "Swing Sell" (reverse logic)

OUTPUT METRICS:
  - Signal type and strength (0-1)
  - Entry price
  - Stop loss (ATR-adjusted)
  - Take profit levels
  - Risk/reward ratio (minimum 1.5x)
  - Technical score (0-100)
  - Timeframe agreement %
  - Volume confirmation flag

EXAMPLE USAGE:
────────────

from swing_scanner import SwingScanner
import pandas as pd

# Load price data with lowercase columns
df = pd.read_csv('stock_data.csv')
df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

# Create scanner
scanner = SwingScanner(min_confidence=60.0)

# Analyze single stock
setup = scanner.scan_swing_setup(df, 'AAPL')

if setup:
    print(f"Signal: {setup.signal}")
    print(f"Entry: ${setup.entry_price:.2f}")
    print(f"Stop: ${setup.stop_loss:.2f}")
    print(f"Target: ${setup.take_profit:.2f}")
    print(f"R/R Ratio: {setup.risk_reward_ratio:.2f}x")
    print(f"Confidence: {setup.confidence:.1f}%")
    print(f"Reasons:")
    for reason in setup.reasons:
        print(f"  • {reason}")

# Analyze multiple stocks
symbols = {
    'AAPL': df_aapl,
    'MSFT': df_msft,
    'TSLA': df_tsla,
}

results = scanner.scan_multiple(symbols)
for setup in results:
    print(scanner.format_setup(setup))

════════════════════════════════════════════════════════════════════════════════

2️⃣ VALUE SCANNER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PURPOSE: Identify undervalued stocks using comprehensive fundamental analysis

KEY FEATURES:
  ✓ Multiple valuation metrics (P/E, P/B, P/S, EV/EBITDA)
  ✓ DCF intrinsic value calculation
  ✓ Quality scoring (ROE, debt, liquidity, FCF quality)
  ✓ Growth analysis (revenue, earnings, FCF growth)
  ✓ Margin of safety calculation
  ✓ Industry comparison benchmarking
  ✓ Investment grade (A+ to D)

VALUATION GRADES:
  • A+ : Deep Value (score 90+)
  • A  : Strong Value (score 80-89)
  • B+ : Moderate Value (score 70-79)
  • B  : Fair Value (score 60-69)
  • C  : Slightly Overvalued (score 40-59)
  • D  : Overvalued (score <40)

OUTPUT METRICS:
  - Overall value score (0-100)
  - Valuation score, quality score, growth score
  - Intrinsic value (DCF estimate)
  - Current price
  - Margin of safety %
  - Key ratios (P/E, P/B, P/S, EV/EBITDA, FCF yield)
  - Quality metrics (ROE, D/E, current ratio)
  - Growth metrics
  - Investment reasons

EXAMPLE USAGE:
────────────

from value_scanner import ValueScanner

scanner = ValueScanner()

# Scan single stock with fundamentals
analysis = scanner.scan_value_stock(
    symbol='AAPL',
    current_price=189.45,
    pe=28.5,
    pb=45.2,
    ps=29.8,
    ev_ebitda=22.1,
    market_cap=3.0e12,
    fcf=110.5e9,
    
    # Quality metrics
    roe=98.5,
    debt_to_equity=0.58,
    current_ratio=1.08,
    
    # Growth
    revenue_growth=7.5,
    earnings_growth=5.2,
    fcf_growth=6.8,
    
    # Industry benchmarks
    industry_pe=20,
    industry_pb=3,
    industry_ps=2,
    fcf_yield_pct=5
)

if analysis:
    print(f"Grade: {analysis.grade}")
    print(f"Score: {analysis.score:.1f}/100")
    print(f"Intrinsic Value: ${analysis.intrinsic_value:.2f}")
    print(f"Current Price: ${analysis.current_price:.2f}")
    print(f"Margin of Safety: {analysis.margin_of_safety:.1f}%")
    print(f"\nKey Metrics:")
    print(f"  P/E: {analysis.pe_ratio:.2f}")
    print(f"  ROE: {analysis.roe:.1f}%")
    print(f"  D/E: {analysis.debt_to_equity:.2f}")
    print(f"  FCF Yield: {analysis.fcf_yield:.1f}%")

════════════════════════════════════════════════════════════════════════════════

3️⃣ MOMENTUM SCANNER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PURPOSE: Identify trending stocks with strong momentum signals

KEY FEATURES:
  ✓ RSI analysis (overbought/oversold)
  ✓ MACD momentum and crossovers
  ✓ Stochastic oscillator
  ✓ Divergence detection (bullish/bearish)
  ✓ Trend confirmation (moving averages)
  ✓ On-balance volume (OBV) analysis
  ✓ Multi-indicator agreement scoring

SIGNALS (5-Star Rating):
  • ⭐⭐⭐⭐⭐ : Strong Buy
  • ⭐⭐⭐⭐  : Buy
  • ⭐⭐⭐   : Neutral
  • ⭐⭐    : Sell
  • ⭐     : Strong Sell

OUTPUT METRICS:
  - Rating (1-5 stars)
  - Momentum score (0-100)
  - Trend strength (0-100)
  - Individual indicator readings (RSI, MACD, Stochastic)
  - Trend direction (Uptrend/Downtrend/Sideways)
  - Divergence flags (bullish/bearish)
  - Multi-indicator agreement %
  - Key signal reasons

EXAMPLE USAGE:
────────────

from momentum_scanner import MomentumScanner
import pandas as pd

scanner = MomentumScanner()

# Analyze DataFrame with price data
analysis = scanner.analyze_momentum(df, 'MSFT')

if analysis:
    stars = "⭐" * analysis.rating
    print(f"{stars}")
    print(f"Signal: {analysis.signal}")
    print(f"Momentum Score: {analysis.momentum_score:.1f}/100")
    print(f"Trend: {analysis.trend_direction}")
    print(f"\nIndicators:")
    print(f"  RSI: {analysis.rsi:.1f} ({analysis.rsi_signal})")
    print(f"  MACD: {analysis.macd_momentum}")
    print(f"  Stochastic: {analysis.stochastic_k:.1f}")
    print(f"\nDivergences:")
    if analysis.bullish_divergence:
        print(f"  ✓ Bullish divergence detected")
    if analysis.bearish_divergence:
        print(f"  ✓ Bearish divergence detected")

# Scan multiple stocks
stocks = {'AAPL': df_aapl, 'MSFT': df_msft, 'GOOGL': df_googl}
results = scanner.scan_stocks(stocks)  # Returns only rating >= 4

for analysis in results:
    print(scanner.format_analysis(analysis))

════════════════════════════════════════════════════════════════════════════════

4️⃣ BREAKOUT SCANNER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PURPOSE: Identify support/resistance breakouts and consolidation patterns

KEY FEATURES:
  ✓ Dynamic support/resistance detection
  ✓ Consolidation range identification
  ✓ Triangle pattern recognition
  ✓ Volume confirmation analysis
  ✓ False breakout risk detection
  ✓ Risk/reward ratio calculation
  ✓ Multi-timeframe confirmation

BREAKOUT TYPES:
  • Range Breakout     - Price breaks out of range
  • Resistance Breakout - Price breaks above resistance
  • Support Breakdown   - Price breaks below support
  • Triangle Breakout   - Price exits triangle pattern
  • Consolidation      - Price consolidating, breakout expected

SIGNALS:
  • "Breakout"       - Broke above resistance with confirmation
  • "Breakdown"      - Broke below support with confirmation
  • "Consolidating"  - In consolidation, waiting for breakout
  • "False Breakout" - High false breakout risk

OUTPUT METRICS:
  - Breakout type and signal
  - Confidence level (0-100)
  - Breakout level (R/S price)
  - Entry price
  - Stop loss and take profit
  - Risk/reward ratio
  - Volume confirmation
  - False breakout risk %
  - Consolidation duration
  - Range size and percentage
  - Technical confirmations

EXAMPLE USAGE:
────────────

from breakout_scanner import BreakoutScanner
import pandas as pd

scanner = BreakoutScanner()

# Analyze single stock
setup = scanner.scan_breakout(df, 'TSLA')

if setup:
    print(f"Type: {setup.breakout_type}")
    print(f"Signal: {setup.signal}")
    print(f"Confidence: {setup.confidence:.1f}%")
    print(f"\nPrice Levels:")
    print(f"  Breakout Level: ${setup.breakout_level:.2f}")
    print(f"  Entry: ${setup.entry_price:.2f}")
    print(f"  Stop: ${setup.stop_loss:.2f}")
    print(f"  Target: ${setup.take_profit:.2f}")
    print(f"  R/R Ratio: {setup.risk_reward_ratio:.2f}x")
    print(f"\nRisk Analysis:")
    print(f"  False Breakout Risk: {setup.false_breakout_risk:.1f}%")
    print(f"  Volume Confirmation: {'Yes' if setup.volume_confirmation else 'No'}")
    print(f"  Volume Ratio: {setup.volume_ratio:.2f}x")
    
# Scan multiple stocks
symbols = {'AAPL': df_aapl, 'MSFT': df_msft}
results = scanner.scan_multiple(symbols)

for setup in sorted(results, key=lambda x: x.confidence, reverse=True):
    print(scanner.format_setup(setup))

════════════════════════════════════════════════════════════════════════════════

5️⃣ UNIFIED SCANNER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PURPOSE: Combine all 4 scanners for consensus signals and high-confidence trades

KEY FEATURES:
  ✓ Multi-strategy consensus analysis
  ✓ Scanner agreement scoring (0-4)
  ✓ Unified signal generation
  ✓ Confidence level calculation
  ✓ Risk level assessment
  ✓ Portfolio-wide scanning
  ✓ Summary reporting

UNIFIED SIGNALS:
  • STRONG BUY (3+ scanners agree - 85% confidence)
  • BUY (2+ scanners agree - 65% confidence)
  • NEUTRAL (mixed signals - 50% confidence)
  • SELL (2+ sellers - 65% confidence)
  • STRONG SELL (3+ sellers - 85% confidence)
  • HOLD (no agreement - 40% confidence)

EXAMPLE USAGE:
────────────

from __init__ import UnifiedTradingScanner
import pandas as pd

# Initialize unified scanner
scanner = UnifiedTradingScanner(
    swing_confidence=60.0,
    value_confidence=60.0,
    momentum_confidence=55.0,
    breakout_confidence=60.0
)

# Scan single stock with all data
scan = scanner.scan_stock(
    symbol='AAPL',
    price_data=df_price,
    fundamental_data={
        'current_price': 189.45,
        'pe': 28.5,
        'pb': 45.2,
        'ps': 29.8,
        'ev_ebitda': 22.1,
        'market_cap': 3.0e12,
        'fcf': 110.5e9,
        'roe': 98.5,
        'debt_to_equity': 0.58,
        'current_ratio': 1.08,
        'revenue_growth': 7.5,
        'earnings_growth': 5.2,
        'fcf_growth': 6.8,
        'industry_pe': 20,
        'industry_pb': 3,
        'industry_ps': 2,
        'fcf_yield_pct': 5
    }
)

# Check results
print(f"Primary Signal: {scan.primary_signal}")
print(f"Overall Score: {scan.overall_score:.1f}/100")
print(f"Confidence: {scan.confidence_level:.1f}%")
print(f"Risk Level: {scan.risk_level}")
print(f"Scanner Agreement: {scan.scanner_agreement}/4")

# Display all scanner results
if scan.swing_setup:
    print(f"\n✓ SWING: {scan.swing_setup.signal}")
if scan.value_analysis:
    print(f"✓ VALUE: Grade {scan.value_analysis.grade}")
if scan.momentum_analysis:
    print(f"✓ MOMENTUM: {scan.momentum_analysis.signal}")
if scan.breakout_setup:
    print(f"✓ BREAKOUT: {scan.breakout_setup.signal}")

# Scan portfolio
portfolio = {
    'AAPL': {'price_data': df_aapl, 'fundamentals': {...}},
    'MSFT': {'price_data': df_msft, 'fundamentals': {...}},
    'GOOGL': {'price_data': df_googl, 'fundamentals': {...}},
    # ... more stocks
}

scans = scanner.scan_portfolio(portfolio)

# Print formatted results
for scan in scans:
    print(scanner.format_scan(scan))

# Generate summary report
report = scanner.generate_report(scans)
print(report)

════════════════════════════════════════════════════════════════════════════════

🔄 WORKFLOW EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WORKFLOW 1: Daily Swing Trade Screening
─────────────────────────────────────────

import pandas as pd
from swing_scanner import SwingScanner

# Load EOD price data
symbols = ['AAPL', 'MSFT', 'TSLA', 'AMZN', 'GOOGL']
scanner = SwingScanner(min_confidence=65)

# Download/load your data (use yfinance, IEX, etc.)
for symbol in symbols:
    df = download_price_data(symbol)  # Your function
    setup = scanner.scan_swing_setup(df, symbol)
    
    if setup and setup.risk_reward_ratio > 2.0:
        print(f"\n{symbol} - {setup.signal}")
        print(f"  Entry: ${setup.entry_price:.2f}")
        print(f"  Target: ${setup.take_profit:.2f}")
        print(f"  Stop: ${setup.stop_loss:.2f}")
        print(f"  R/R: {setup.risk_reward_ratio:.2f}x")


WORKFLOW 2: Value Stock Discovery
─────────────────────────────────

from value_scanner import ValueScanner

scanner = ValueScanner()

# Get list of candidates (e.g., from screening or watchlist)
candidates = [
    {'symbol': 'GE', 'pe': 8.5, 'pb': 0.8, ...},  # Your fundamental data
    {'symbol': 'DISH', 'pe': 7.2, 'pb': 0.6, ...},
    # ... more candidates
]

value_stocks = []

for stock in candidates:
    analysis = scanner.scan_value_stock(**stock)
    if analysis and analysis.grade in ['A+', 'A']:
        value_stocks.append(analysis)

# Sort by margin of safety
value_stocks.sort(key=lambda x: x.margin_of_safety, reverse=True)

for stock in value_stocks[:10]:
    print(f"{stock.symbol:6} - {stock.grade} │ MOS: {stock.margin_of_safety:6.1f}%")


WORKFLOW 3: Momentum-Based Trading
──────────────────────────────────

from momentum_scanner import MomentumScanner

scanner = MomentumScanner()

# Scan all holdings for momentum changes
portfolio_symbols = ['AAPL', 'MSFT', 'NVDA', 'META', 'AMZN']
strong_signals = []

for symbol in portfolio_symbols:
    df = get_price_data(symbol)
    analysis = scanner.analyze_momentum(df, symbol)
    
    if analysis and analysis.rating >= 4:
        strong_signals.append(analysis)

# Alert on strong buy signals
for analysis in strong_signals:
    print(f"⭐⭐⭐⭐⭐ {analysis.symbol}: {analysis.signal}")
    print(f"   Momentum Score: {analysis.momentum_score:.0f}")
    print(f"   Trend: {analysis.trend_direction}")


WORKFLOW 4: Breakout Trade Setup
────────────────────────────────

from breakout_scanner import BreakoutScanner

scanner = BreakoutScanner(min_confidence=65)

# Monitor stocks in consolidation
watched = ['TSLA', 'NVDA', 'AMD', 'CRM']
trades = []

for symbol in watched:
    df = get_latest_data(symbol)
    setup = scanner.scan_breakout(df, symbol)
    
    if setup and setup.signal == "Breakout" and not setup.false_breakout_risk > 50:
        trades.append(setup)

# Prepare trade execution
for setup in sorted(trades, key=lambda x: x.risk_reward_ratio, reverse=True):
    print(f"\n📊 {setup.symbol} - {setup.breakout_type}")
    print(f"Signal: {setup.signal} ({setup.confidence:.0f}% confidence)")
    print(f"Entry: ${setup.entry_price:.2f} | Stop: ${setup.stop_loss:.2f} | Target: ${setup.take_profit:.2f}")
    print(f"Risk/Reward: {setup.risk_reward_ratio:.2f}x")


WORKFLOW 5: Complete Analysis Pipeline
──────────────────────────────────────

from __init__ import UnifiedTradingScanner
import pandas as pd

# Initialize unified scanner
scanner = UnifiedTradingScanner()

# Load stocks and data
stocks_to_analyze = ['AAPL', 'MSFT', 'TSLA']  # Your list

portfolio_data = {}
for symbol in stocks_to_analyze:
    portfolio_data[symbol] = {
        'price_data': load_price_data(symbol),
        'fundamentals': load_fundamentals(symbol)
    }

# Run unified scan
results = scanner.scan_portfolio(portfolio_data)

# Filter for high-confidence signals
strong_buys = [s for s in results if s.primary_signal == 'STRONG BUY']
buys = [s for s in results if s.primary_signal == 'BUY']

print(f"\nStrong Buys ({len(strong_buys)}):")
for scan in strong_buys:
    print(f"  {scan.symbol:6} - Confidence: {scan.confidence_level:.0f}%")

print(f"\nBuys ({len(buys)}):")
for scan in buys:
    print(f"  {scan.symbol:6} - Confidence: {scan.confidence_level:.0f}%")

# Print detailed report
print("\n" + scanner.generate_report(results))

════════════════════════════════════════════════════════════════════════════════

📊 PERFORMANCE TUNING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ADJUST CONFIDENCE LEVELS:

Higher confidence = fewer signals, higher quality
Lower confidence = more signals, higher false positive rate

Recommended starting values:
  - Swing Scanner:   60-65%
  - Value Scanner:   60-70%
  - Momentum Scanner: 55-60%
  - Breakout Scanner: 60-70%

OPTIMIZE FOR YOUR STRATEGY:

Day Trading:
  • Swing: 60% | Momentum: 55% | Breakout: 60%
  • Focus: short-term patterns, high volume confirmation

Swing Trading:
  • Swing: 65% | Value: N/A | Momentum: 60% | Breakout: 65%
  • Focus: 3-10 day trends, support/resistance bounces

Value Investing:
  • Value: 70% | Fundamentals: strong
  • Focus: undervalued picks, long-term holds

Growth Investing:
  • Momentum: 60% | Value: Growth focus | Breakout: 65%
  • Focus: trending stocks, breakouts to new highs

════════════════════════════════════════════════════════════════════════════════

✅ TESTING & VALIDATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BACKTEST YOUR SCANNERS:

import pandas as pd
from __init__ import UnifiedTradingScanner

# Load historical data
df = pd.read_csv('historical_data.csv')

scanner = UnifiedTradingScanner()

# Test on past data
results = []
for row_idx in range(100, len(df), 5):  # Every 5 days
    recent_data = df.iloc[row_idx-100:row_idx]
    scan = scanner.scan_stock('TEST', recent_data)
    results.append(scan)

# Analyze results
strong_buys = [r for r in results if r.primary_signal == 'STRONG BUY']
win_rate = calculate_returns(strong_buys)  # Your function

print(f"Strong Buy Signals: {len(strong_buys)}")
print(f"Average Win Rate: {win_rate:.1f}%")

════════════════════════════════════════════════════════════════════════════════

🚀 INTEGRATION WITH STOCKAI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Combine with StockAI for comprehensive analysis:

from __init__ import StockAnalysisEngine, UnifiedTradingScanner
import pandas as pd

# Initialize both systems
ai_engine = StockAnalysisEngine()
trade_scanner = UnifiedTradingScanner()

# Analyze with both systems
symbol = 'AAPL'
df = load_data(symbol)

# StockAI analysis (fundamentals, technicals, peers)
ai_analysis = ai_engine.analyze_stock(symbol)

# Trading signals (swing, value, momentum, breakout)
trade_scan = trade_scanner.scan_stock(symbol, df, fundamentals)

# Combine insights
print(f"StockAI Score: {ai_analysis.technical['volatility']}")
print(f"Trading Signal: {trade_scan.primary_signal}")
print(f"Confidence: {trade_scan.confidence_level:.0f}%")

═══════════════════════════════════════════════════════════════════════════════════

📈 SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUICK REFERENCE:

SWING SCANNER:
  Use for: Short-term trades (3-10 days)
  Signal: Pattern + momentum confirmation
  Best with: Technical analysis skills

VALUE SCANNER:
  Use for: Long-term value picks
  Signal: Undervalued metrics + margin of safety
  Best with: Fundamental analysis skills

MOMENTUM SCANNER:
  Use for: Trend-following trades
  Signal: Multiple indicator agreement
  Best with: Trend-trading experience

BREAKOUT SCANNER:
  Use for: Support/resistance breakouts
  Signal: Volume + pattern confirmation
  Best with: Level-based trading strategies

UNIFIED SCANNER:
  Use for: High-confidence consensus signals
  Signal: 3+ scanners in agreement
  Best with: All trading styles (most reliable)

════════════════════════════════════════════════════════════════════════════════

For complete API reference, see individual scanner module docstrings.
For examples and use cases, see QUICK_START.txt and IMPLEMENTATION_GUIDE.txt.

Good luck trading! 📊🚀
