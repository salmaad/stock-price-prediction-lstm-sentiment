import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import feedparser
import time

# ---------- Setup ----------
nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()

st.set_page_config(page_title="AI Stock Predictor", page_icon="📈", layout="wide")
st.title("📊 Stock Price Prediction with News Sentiment (LSTM)")
st.write("This app combines **stock price trends** with **news sentiment** to predict future prices using an LSTM model.")

# ---------- Sidebar inputs ----------
ticker_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, MSFT):", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
train_ratio = st.sidebar.slider("Train/Test Split Ratio", 0.5, 0.95, 0.8)
time_step = 60
epochs = st.sidebar.slider("Epochs", 5, 50, 15)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1)

# ---------- Helpers: News with multiple fallbacks ----------
def fetch_news_yf_ticker(sym: str):
    rows = []
    try:
        t = yf.Ticker(sym)
        n = t.news or []
        for it in n:
            title = it.get("title") or it.get("content") or it.get("summary")
            pub = pd.to_datetime(it.get("providerPublishTime"), unit="s", errors="coerce")
            if title and pd.notna(pub):
                rows.append({"title": str(title), "publisher": it.get("publisher", ""), "published": pub})
    except Exception:
        pass
    return rows

def fetch_news_yf_global(sym: str):
    rows = []
    try:
        from yfinance import get_news
        n = get_news(sym) or []
        for it in n:
            title = it.get("title") or it.get("content") or it.get("summary")
            pub = pd.to_datetime(it.get("providerPublishTime"), unit="s", errors="coerce")
            if title and pd.notna(pub):
                rows.append({"title": str(title), "publisher": it.get("publisher", ""), "published": pub})
    except Exception:
        pass
    return rows

def fetch_news_google_rss(sym: str):
    rows = []
    try:
        # Google News RSS query for "<ticker> stock"
        q = f"https://news.google.com/rss/search?q={sym}+stock&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(q)
        for e in feed.entries:
            title = getattr(e, "title", None)
            # published_parsed may be a time.struct_time
            pub_struct = getattr(e, "published_parsed", None)
            pub = pd.to_datetime(time.strftime("%Y-%m-%d %H:%M:%S", pub_struct)) if pub_struct else pd.NaT
            publisher = getattr(e, "source", {}).get("title", "") if hasattr(e, "source") else ""
            if title and pd.notna(pub):
                rows.append({"title": str(title), "publisher": publisher, "published": pub})
    except Exception:
        pass
    return rows

def fetch_news_finviz(sym: str):
    rows = []
    try:
        from finvizfinance.quote import finvizfinance as FinvizQuote
        fq = FinvizQuote(sym)
        # Some versions use ticker_news, others TickerNews; try both safely
        news_df = None
        try:
            news_df = fq.ticker_news()
        except Exception:
            try:
                news_df = fq.TickerNews()
            except Exception:
                news_df = None
        if isinstance(news_df, pd.DataFrame) and not news_df.empty:
            # Expected columns: ['Date','Title','Link'] (Date like 'Oct-31-24 04:45PM')
            for _, r in news_df.iterrows():
                title = r.get("Title")
                pub = pd.to_datetime(r.get("Date"), errors="coerce")
                if title and pd.notna(pub):
                    rows.append({"title": str(title), "publisher": "Finviz", "published": pub})
    except Exception:
        pass
    return rows

def fetch_news_df(sym: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.DataFrame:
    rows = []
    rows += fetch_news_yf_ticker(sym)
    if not rows:
        rows += fetch_news_yf_global(sym)
    if not rows:
        rows += fetch_news_google_rss(sym)
    if not rows:
        rows += fetch_news_finviz(sym)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Normalize / filter
    df["published"] = pd.to_datetime(df["published"], errors="coerce")
    df = df.dropna(subset=["published"])
    df = df[(df["published"].dt.date >= pd.to_datetime(start_dt).date()) &
            (df["published"].dt.date <= pd.to_datetime(end_dt).date())]
    df = df.sort_values("published", ascending=False).drop_duplicates(subset=["title"])
    return df

def sentiment_by_day(news_df: pd.DataFrame) -> pd.DataFrame:
    if news_df.empty:
        return pd.DataFrame(columns=["Date", "Sentiment"])
    df = news_df.copy()
    df["title"] = df["title"].astype(str)
    df["sentiment"] = df["title"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
    df["published"] = pd.to_datetime(df["published"], errors="coerce")
    df = df.dropna(subset=["published"])
    if df.empty:
        return pd.DataFrame(columns=["Date", "Sentiment"])
    daily = (
        df.groupby(df["published"].dt.date)["sentiment"]
        .mean()
        .reset_index()
        .rename(columns={"published": "Date", "sentiment": "Sentiment"})
    )
    return daily

# ---------- Fetch price data ----------
st.subheader(f"Stock Data for {ticker_symbol}")
ticker = yf.Ticker(ticker_symbol)
data = ticker.history(start=start_date, end=end_date)

if data.empty:
    st.error("No data found for this ticker symbol. Please check it and try again.")
    st.stop()

st.line_chart(data["Close"])

# ---------- Fetch & show news ----------
st.subheader("Recent News Sentiment (multiple free sources)")
news_df = fetch_news_df(ticker_symbol, start_date, end_date)
if news_df.empty:
    st.warning("No news found from Yahoo/Google RSS/Finviz for this period/ticker. Training will use sentiment=0.")
    st.dataframe(pd.DataFrame(columns=["title", "publisher", "published", "sentiment"]))
else:
    tmp = news_df.copy()
    tmp["sentiment"] = tmp["title"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
    st.dataframe(tmp[["title", "publisher", "published", "sentiment"]].head(15))

news_sentiment = sentiment_by_day(news_df)

# ---------- Merge price + sentiment ----------
df = data[["Close"]].reset_index()
df["Date"] = pd.to_datetime(df["Date"]).dt.date
merged = pd.merge(df, news_sentiment, on="Date", how="left")
merged["Sentiment"] = merged["Sentiment"].fillna(0.0)

st.subheader("Merged Dataset (Close + Daily Sentiment)")
st.dataframe(merged.tail(12))

# ---------- Prepare data ----------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(merged[["Close", "Sentiment"]])

train_size = int(len(scaled_data) * train_ratio)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size - time_step:]

def create_dataset(dataset, step=60):
    X, y = [], []
    for i in range(step, len(dataset)):
        X.append(dataset[i - step:i])   # shape: (step, features)
        y.append(dataset[i, 0])         # predict normalized 'Close'
    return np.array(X), np.array(y)

X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape for LSTM [samples, time_steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 2)
X_test  = X_test.reshape(X_test.shape[0],  X_test.shape[1],  2)

# ---------- Build model ----------
model = Sequential()
model.add(LSTM(96, return_sequences=True, input_shape=(X_train.shape[1], 2)))
model.add(Dropout(0.3))
model.add(LSTM(96, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(48, activation="relu"))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mean_squared_error")

# ---------- Train ----------
st.write("Training LSTM model with price + sentiment features...")
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

# ========== PREDICTIONS ==========
predicted = model.predict(X_test)
predicted_prices = scaler.inverse_transform(
    np.hstack((predicted, np.zeros_like(predicted)))
)[:, 0]

actual_prices = scaler.inverse_transform(
    np.hstack((y_test.reshape(-1, 1), np.zeros_like(y_test.reshape(-1, 1))))
)[:, 0]

# ========== PLOT ACTUAL vs PREDICTED ==========
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(actual_prices, color='blue', label='Actual Prices')
ax1.plot(predicted_prices, color='red', label='Predicted Prices')
ax1.set_title(f"{ticker_symbol} — Actual vs Predicted Closing Prices")
ax1.set_xlabel("Days")
ax1.set_ylabel("Price")
ax1.legend()
st.pyplot(fig1)

# ========== 3-MONTH FORECAST ==========
forecast_days = 60  # roughly 3 months of trading days
last_window = scaled_data[-time_step:].copy()
future_predictions = []

for _ in range(forecast_days):
    X_future = np.array([last_window]).reshape(1, time_step, 2)
    next_norm = model.predict(X_future, verbose=0)
    # assume sentiment stays neutral (0)
    next_input = np.array([[next_norm[0][0], 0.0]])
    last_window = np.vstack((last_window[1:], next_input))
    future_predictions.append(next_norm[0][0])

# inverse transform predictions
future_prices = scaler.inverse_transform(
    np.hstack((np.array(future_predictions).reshape(-1, 1), np.zeros((forecast_days, 1))))
)[:, 0]

# ========== PLOT FORECAST ==========
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(np.arange(len(merged)), merged['Close'], label='Historical Prices', color='blue')
ax2.plot(np.arange(len(merged), len(merged) + forecast_days), future_prices, label='Forecast (3 months)', color='orange')
ax2.set_title(f"{ticker_symbol} — 3-Month Price Forecast")
ax2.set_xlabel("Days")
ax2.set_ylabel("Price")
ax2.legend()
st.pyplot(fig2)

# ========== SHOW NEXT-DAY PRICE ==========
st.subheader("📈 Predicted Next Day Closing Price:")
st.success(f"${future_prices[0]:.2f}")
