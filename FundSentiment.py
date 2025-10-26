import yfinance as yf
import pandas as pd
from tqdm import tqdm
import requests

# ---------------------------
# Config
# ---------------------------
tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "JPM", "XOM", "NVDA"]
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
HUGGINGFACE_API_KEY = "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"  # Replace with your token

# ---------------------------
# Helper: Hugging Face FinBERT Sentiment
# ---------------------------
def get_sentiment(text):
    """
    Returns a sentiment score between 0 and 1:
    - 0 = negative
    - 0.5 = neutral
    - 1 = positive
    """
    try:
        response = requests.post(
            HUGGINGFACE_API_URL,
            headers={"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"},
            json={"inputs": text}
        )
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            # Pick the label with the highest probability
            best = max(data[0], key=lambda x: x['score'])
            label = best['label'].lower()
            score = best['score']
            if label == "positive":
                return score
            elif label == "negative":
                return 1 - score
            else:  # neutral
                return 0.5
        return 0.5
    except Exception as e:
        print(f"Error processing text: {e}")
        return 0.5

# ---------------------------
# Fetch stock news headlines for sentiment
# ---------------------------
def fetch_news_headlines(ticker):
    """
    Fetches latest news headlines for a ticker using yfinance.
    """
    try:
        stock = yf.Ticker(ticker)
        news_items = stock.news
        headlines = [item['title'] for item in news_items if 'title' in item]
        if len(headlines) == 0:
            headlines = [f"{ticker} latest news"]  # fallback
        return headlines
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        return [f"{ticker} latest news"]

# ---------------------------
# Compute sentiment score per ticker
# ---------------------------
def compute_sentiment_scores(tickers):
    df_scores = pd.DataFrame(columns=["ticker", "sentiment_score"])
    for ticker in tqdm(tickers, desc="Fetching Sentiment Data"):
        headlines = fetch_news_headlines(ticker)
        scores = [get_sentiment(h) for h in headlines]
        avg_score = sum(scores) / len(scores)
        df_scores = pd.concat([df_scores, pd.DataFrame({"ticker":[ticker], "sentiment_score":[avg_score]})], ignore_index=True)
    return df_scores

# ---------------------------
# Example: merge with your buy_score
# ---------------------------
# Dummy buy_score for demonstration; replace with your actual computation
buy_scores = pd.DataFrame({
    "ticker": tickers,
    "buy_score": [0.75, 0.65, 0.80, 0.55, 0.60, 0.50, 0.70]
})

# Compute sentiment scores
df_sentiment = compute_sentiment_scores(tickers)

# Merge
df_combined = buy_scores.merge(df_sentiment, on="ticker")

# Compute final buy score: weighted average (example: 70% buy_score, 30% sentiment)
df_combined["final_buy_score"] = df_combined["buy_score"]*0.7 + df_combined["sentiment_score"]*0.3

# Assign simple rating
def buy_rating(score):
    if score >= 0.75:
        return "Strong Buy"
    elif score >= 0.6:
        return "Buy"
    elif score >= 0.45:
        return "Hold"
    else:
        return "Sell"

df_combined["buy_rating"] = df_combined["final_buy_score"].apply(buy_rating)

# Display results
print("\nFinal Combined Buy Scores with Sentiment:")
print(df_combined)
