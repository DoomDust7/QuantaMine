# app.py
import streamlit as st
import asyncio
import pandas as pd
import yfinance as yf
from fundfinal2 import run_analysis   # import your backend async function

# -------------------------------
# Page setup
# -------------------------------
st.set_page_config(page_title="AI Financial Analyst", layout="wide")

st.title("💹 QuantaMine")
st.markdown("""
Enter one or more **Yahoo Finance tickers** separated by commas.  
Examples:  
- US: `AAPL, MSFT, NVDA, TSLA, JPM`
- ETFs: `SPY, QQQ`
- International: `RELIANCE.NS, BMW.DE, 7203.T`
""")

# -------------------------------
# User input
# -------------------------------
user_input = st.text_input(
    "Tickers",
    placeholder="AAPL, MSFT, NVDA, TSLA",
    help="Type one or more Yahoo Finance tickers separated by commas"
)

tickers = [t.strip().upper() for t in user_input.split(",") if t.strip()]

# -------------------------------
# Helper to validate tickers
# -------------------------------
def validate_tickers(tickers):
    """Check if each ticker exists in Yahoo Finance."""
    valid = []
    invalid = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            if not info or info is None or info == {}:
                invalid.append(t)
            else:
                valid.append(t)
        except Exception:
            invalid.append(t)
    return valid, invalid

# -------------------------------
# Run button
# -------------------------------
if st.button("Run Analysis"):
    if not tickers:
        st.warning("⚠️ Please enter at least one ticker symbol.")
    else:
        valid_tickers, invalid_tickers = validate_tickers(tickers)

        if invalid_tickers:
            st.error(f"❌ Invalid ticker(s): {', '.join(invalid_tickers)}")
            st.stop()

        st.info(f"Running analysis for: {', '.join(valid_tickers)}")

        with st.spinner("Fetching data and generating insights ⏳"):
            try:
                df = asyncio.run(run_analysis(valid_tickers))
                st.success("✅ Analysis complete!")

                st.subheader("📊 Final Results")
                st.dataframe(
                    df[[
                        "ticker",
                        "final_buy_score", "final_rating",
                        "buy_score", "sentiment_score", "llm_confidence",
                        "value_score", "quality_score", "risk_score", "momentum_score",
                        "llm_reason"
                    ]].sort_values("final_buy_score", ascending=False),
                    use_container_width=True
                )

                # Download button
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Results as CSV",
                    csv,
                    "ai_stock_analysis.csv",
                    "text/csv"
                )
            except Exception as e:
                st.error(f"⚠️ Error while processing tickers:\n\n{e}")
