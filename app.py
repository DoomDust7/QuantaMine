# app.py
import io
import asyncio
from datetime import datetime
from typing import List, Tuple

import pandas as pd
import streamlit as st
import yfinance as yf
from reportlab.lib.pagesizes import letter
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, Spacer, SimpleDocTemplate, Table, TableStyle
import reportlab.lib.colors as colors

# Try to import a top-level run_analysis (quantamine) if available; otherwise we'll fall back to analytics primitives.
try:
    from quantamine import run_analysis as top_run_analysis
except Exception:
    top_run_analysis = None

# Import analytics functions (used by fallback pipeline)
try:
    from analytics.fundamentals import get_bulk_fundamentals
    from analytics.risk import get_bulk_risk
    from analytics.sentiment import sentiment_for_ticker
    from analytics.scoring import compute_value_quality_scores
    from analytics.momentum import compute_risk_momentum_scores
    from analytics.llm import run_batched_llm
    ANALYTICS_AVAILABLE = True
except Exception as e:
    print(f"Analytics import failed: {e}")
    ANALYTICS_AVAILABLE = False

st.set_page_config(page_title="QuantaMine - AI Financial Analyst", layout="wide")

st.markdown("""
<style>
    .main-header {font-size: 2.6rem !important; font-weight: bold; color: #1E90FF;}
    .score-good {color: green; font-weight: bold;}
    .score-bad {color: red; font-weight: bold;}
    .stButton>button {background-color: #1E90FF; color: white;}
    .footer {text-align: center; margin-top: 50px; color: #888; font-size: 0.8rem;}
</style>
""", unsafe_allow_html=True)


# -------------------------------
# Helper Functions
# -------------------------------
@st.cache_data(ttl=3600)
def validate_tickers_cached(tickers: List[str]) -> Tuple[List[str], List[str]]:
    valid, invalid = [], []
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            info = tk.info or {}
            symbol = info.get("symbol") or info.get("shortName") or info.get("regularMarketPrice")
            if symbol:
                valid.append(t)
            else:
                invalid.append(t)
        except Exception:
            invalid.append(t)
    return valid, invalid


def create_pdf_report(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("QuantaMine AI Financial Analysis", styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    pdf_df = df.copy()
    if "llm_reason" in pdf_df.columns:
        pdf_df["llm_reason"] = pdf_df["llm_reason"].astype(str).apply(lambda x: x.replace("\n", " ")[:140])

    table_data = [list(pdf_df.columns)]
    for _, row in pdf_df.iterrows():
        row_vals = []
        for v in row.tolist():
            try:
                row_vals.append("" if pd.isna(v) else str(v))
            except Exception:
                row_vals.append(str(v))
        table_data.append(row_vals)

    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    elements.append(table)
    doc.build(elements)
    return buffer.getvalue()


# -------------------------------
# UI: Left column (inputs)
# -------------------------------
col1, col2 = st.columns([1, 2])
with col1:
    st.markdown("<h1 class='main-header'>QuantaMine</h1>", unsafe_allow_html=True)
    st.markdown("**AI-Powered Stock Analysis Engine**")

    popular_tickers = {
        "FAANG": ["AAPL", "AMZN", "META", "GOOG", "NFLX"],
        "Magnificent 7": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META"],
        "ETFs": ["SPY", "QQQ", "IWM", "GLD"],
        "Banks": ["JPM", "BAC", "WFC"],
        "International": ["RELIANCE.NS", "BMW.DE", "7203.T"],
    }

    selected_group = st.selectbox("Or pick a preset:", ["Custom"] + list(popular_tickers.keys()))
    default_tickers = popular_tickers.get(selected_group, []) if selected_group != "Custom" else []

    user_input = st.text_input(
        "Enter Tickers (comma-separated)",
        value=", ".join(default_tickers),
        placeholder="AAPL, MSFT, NVDA",
        help="Examples: AAPL, TSLA, 7203.T, RELIANCE.NS",
    )

    raw_tickers = [t.strip().upper() for t in user_input.split(",") if t.strip()]
    tickers = list(dict.fromkeys(raw_tickers))

    if tickers:
        with st.spinner("Validating tickers..."):
            valid_tickers, invalid_tickers = validate_tickers_cached(tickers)

        if invalid_tickers:
            st.error(f"Invalid: {', '.join(invalid_tickers)}")
        if valid_tickers:
            st.success(
                f"Valid: {', '.join(valid_tickers[:10])}{'...' if len(valid_tickers) > 10 else ''}"
            )

        if len(valid_tickers) > 15:
            st.warning("More than 15 tickers may slow down analysis.")

    years = st.slider("Historical Data (Years)", 1, 10, 5)
    run_button = st.button("🚀 Run Analysis", type="primary")
    enable_llm = st.checkbox("Enable LLM (LLM predictions / Sentiment)", value=False)


# -------------------------------
# Right column: Run & Results
# -------------------------------
with col2:
    if run_button:
        if not tickers:
            st.warning("Please enter at least one ticker.")
        elif "valid_tickers" not in locals() or not valid_tickers:
            st.warning("No valid tickers to analyze.")
        else:
            st.info(f"Analyzing: {', '.join(valid_tickers)}")
            progress = st.progress(0)
            status = st.empty()

            # -------------------------------
            # Fallback pipeline with optional LLM
            # -------------------------------
            async def fallback_pipeline(tickers_list):
                status.text("Fetching fundamentals...")
                progress.progress(10)
                fund_df = await get_bulk_fundamentals(tickers_list)

                status.text("Fetching risk & historical data...")
                progress.progress(30)
                risk_rows = await get_bulk_risk(tickers_list, years=years)
                risk_df = pd.DataFrame(risk_rows)

                status.text("Computing momentum features...")
                progress.progress(45)
                from analytics.momentum import compute_momentum_features

                momentum_rows = []
                for t in tickers_list:
                    try:
                        hist = yf.Ticker(t).history(period=f"{years}y")['Close']
                        feats = compute_momentum_features(hist)
                        feats["ticker"] = t
                        momentum_rows.append(feats)
                    except Exception:
                        momentum_rows.append({"ticker": t, "dist_52w_high": 0.0, "sma50_dist": 0.0, "sma200_dist": 0.0})
                momentum_df = pd.DataFrame(momentum_rows)

                status.text("Computing sentiment...")
                progress.progress(55)

                # Use LLM in sentiment only if enable_llm=True
                if enable_llm:
                    sent_tasks = [sentiment_for_ticker(t) for t in tickers_list]
                else:
                    async def neutral_sentiment(ticker):
                        return {"ticker": ticker, "sentiment_score": 0.5}
                    sent_tasks = [neutral_sentiment(t) for t in tickers_list]

                sent_rows = await asyncio.gather(*sent_tasks, return_exceptions=False)
                sent_df = pd.DataFrame(sent_rows)

                status.text("Scoring fundamentals...")
                progress.progress(65)
                fund_scored = compute_value_quality_scores(fund_df)

                status.text("Scoring risk & momentum...")
                progress.progress(75)
                risk_scored = compute_risk_momentum_scores(risk_df, momentum_df)

                status.text("Merging all data...")
                progress.progress(80)
                merged = fund_scored.merge(risk_scored, on="ticker", how="left").merge(sent_df, on="ticker", how="left")

                # Ensure all pillars exist
                for col in ["value_score", "quality_score", "risk_score", "momentum_score", "sentiment_score"]:
                    if col not in merged.columns:
                        merged[col] = 2.5
                merged[["value_score", "quality_score", "risk_score", "momentum_score", "sentiment_score"]] = \
                    merged[["value_score", "quality_score", "risk_score", "momentum_score", "sentiment_score"]].fillna(2.5)

                # Optional LLM explanations
                merged["llm_confidence"] = 0.5
                merged["llm_reason"] = "LLM disabled"
                if enable_llm:
                    status.text("Running batched LLM analysis...")
                    progress.progress(85)
                    rows_for_llm = merged[["ticker", "value_score", "quality_score", "risk_score",
                                           "momentum_score", "sentiment_score"]].to_dict(orient="records")
                    llm_results = await run_batched_llm(rows_for_llm)
                    merged["llm_confidence"] = merged["ticker"].apply(lambda t: llm_results.get(t, {}).get("confidence", 0.5))
                    merged["llm_reason"] = merged["ticker"].apply(lambda t: llm_results.get(t, {}).get("reason", "No reason"))

                # Compute final buy score
                def final_buy_score_from_pillars(r):
                    w_value, w_quality, w_risk, w_momentum, w_sentiment = 0.25, 0.25, 0.20, 0.20, 0.10
                    agg5 = (w_value * r["value_score"] + w_quality * r["quality_score"] + w_risk * r["risk_score"] +
                            w_momentum * r["momentum_score"] + w_sentiment * r["sentiment_score"])
                    llm5 = r.get("llm_confidence", 0.5) * 5.0
                    blend = 0.6
                    final5 = blend * agg5 + (1 - blend) * llm5
                    return max(0.0, min(1.0, final5 / 5.0))

                merged["final_buy_score"] = merged.apply(final_buy_score_from_pillars, axis=1)

                def buy_rating(score):
                    if score >= 0.75: return "Strong Buy"
                    elif score >= 0.65: return "Buy"
                    elif score >= 0.45: return "Hold"
                    else: return "Avoid"

                merged["final_rating"] = merged["final_buy_score"].apply(buy_rating)

                for c in ["value_score", "quality_score", "risk_score", "momentum_score", "sentiment_score"]:
                    merged[c] = merged[c].round(2)
                merged["final_buy_score"] = merged["final_buy_score"].round(4)
                merged["llm_confidence"] = merged["llm_confidence"].round(3)

                progress.progress(100)
                status.text("Complete ✅")
                return merged

            # -------------------------------
            # Main runner
            # -------------------------------
            try:
                progress.progress(3)
                status.text("Preparing analysis...")

                if top_run_analysis is not None:
                    try:
                        result = top_run_analysis(valid_tickers, history_years=years, enable_llm=enable_llm)
                    except TypeError:
                        result = top_run_analysis(valid_tickers)

                    if asyncio.iscoroutine(result):
                        df = asyncio.run(result)
                    else:
                        if asyncio.iscoroutinefunction(top_run_analysis):
                            df = asyncio.run(top_run_analysis(valid_tickers, history_years=years, enable_llm=enable_llm))
                        else:
                            df = result
                else:
                    df = asyncio.run(fallback_pipeline(valid_tickers))

                if df is None:
                    raise RuntimeError("Analysis returned no results.")
                if not isinstance(df, pd.DataFrame):
                    df = pd.DataFrame(df)

                # Ensure expected columns
                expected_cols = ["ticker", "final_buy_score", "final_rating", "value_score",
                                 "quality_score", "risk_score", "momentum_score", "sentiment_score",
                                 "llm_confidence", "llm_reason"]
                for c in expected_cols:
                    if c not in df.columns:
                        if c in ["llm_reason", "final_rating"]:
                            df[c] = ""
                        elif c == "llm_confidence":
                            df[c] = 0.5
                        else:
                            df[c] = pd.NA

                for c in ["final_buy_score", "value_score", "quality_score", "risk_score",
                          "momentum_score", "sentiment_score", "llm_confidence"]:
                    if c in df.columns:
                        try:
                            df[c] = pd.to_numeric(df[c], errors="coerce").round(4)
                        except Exception:
                            pass

                if df.empty:
                    st.warning("No results produced.")
                else:
                    result_cols = ["ticker", "final_buy_score", "final_rating", "sentiment_score",
                                   "llm_confidence", "value_score", "quality_score", "risk_score",
                                   "momentum_score", "llm_reason"]
                    result_cols = [c for c in result_cols if c in df.columns]
                    display_df = df[result_cols].copy().sort_values("final_buy_score", ascending=False)

                    st.subheader("📊 AI Stock Ratings")
                    st.dataframe(display_df, use_container_width=True)

                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        csv = display_df.to_csv(index=False).encode("utf-8")
                        st.download_button("📄 Download CSV", csv, "quantamine_analysis.csv", "text/csv")
                    with col_dl2:
                        try:
                            pdf_bytes = create_pdf_report(display_df)
                            st.download_button("📑 Download PDF Report", pdf_bytes, "quantamine_report.pdf", "application/pdf")
                        except Exception as e:
                            st.error(f"Could not generate PDF: {e}")

                    with st.expander("What do these scores mean?"):
                        st.markdown("""
- **final_buy_score**: 0–1 AI confidence in buying (higher = stronger buy)
- **final_rating**: Human-readable bucket from final score
- **value_score**: Valuation metrics (PE, PB, PEG, EV/EBITDA, dividend)
- **quality_score**: Profitability, growth, FCF yield, leverage
- **risk_score**: Volatility, drawdown, Sharpe
- **momentum_score**: Recent returns + RSI
- **sentiment_score**: Aggregated news sentiment (FinBERT + multi-source)
- **llm_confidence**: LLM's buy confidence (if enabled)
""")
            except Exception as e:
                status.empty()
                progress.empty()
                st.error(f"Analysis failed: {e}")
                st.exception(e)

st.markdown("<div class='footer'>Powered by xAI, yfinance, and Streamlit | Not financial advice</div>", unsafe_allow_html=True)
