# create a stock scoring system combining
# RSI, MACD, volume breakout, and fundamentals
import sys
import os
import yfinance as yf

# ensure Python can find local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from technical.trend_detection import detect_trend
from technical.breakout import breakout_signal
from technical.support_resistance import get_support_resistance
from technical.volume_profile import volume_profile

from ai.price_prediction import predict_price
from ai.signal_model import ai_signal
from ai.ranking_engine import stock_score


symbol = input("Enter Stock Symbol: ")

df = yf.download(symbol, period="1y", interval="1d")

trend = detect_trend(df)
breakout = breakout_signal(df)
support, resistance = get_support_resistance(df)
volume = volume_profile(df)

prediction = predict_price(df)
ai = ai_signal(df)
score = stock_score(df)

if score >= 80:
    signal = "STRONG BUY"
elif score >= 60:
    signal = "BUY"
elif score >= 40:
    signal = "HOLD"
else:
    signal = "SELL"

print("\n====== STOCK ANALYSIS ======")

print("Stock:", symbol)
print("Trend:", trend)
print("Breakout:", breakout)

print("\nSupport:", support)
print("Resistance:", resistance)

print("\nVolume Profile:", volume)

print("\nAI Prediction:", prediction)
print("AI Signal:", ai)

print("\nAI Score:", score, "/100")