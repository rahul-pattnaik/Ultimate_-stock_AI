from transformers import pipeline

sentiment = pipeline("sentiment-analysis")

def analyze_news(text):

    result = sentiment(text)

    return result[0]["label"]