
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Price Prediction Dashboard")
st.write("Explore historical stock prices and predict the next available trading day's closing price.")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Amazon stock Data.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

try:
    df = load_data()
except Exception as e:
    st.error("Could not load Amazon stock Data.csv. Put it in the same folder as app.py.")
    st.stop()

# Amazon dataset: only one stock, so no Name column is needed
df.columns = df.columns.str.strip().str.lower()

# Remove accidental spaces and blank rows
df = df.dropna(how="all").copy()

df["date"] = pd.to_datetime(df["date"], errors="coerce")

for col in ["open", "high", "low", "close", "volume"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

stock_df = (
    df.dropna(subset=["date", "close"])
    .sort_values("date")
    .drop_duplicates(subset="date", keep="last")
    .copy()
)

stock = "AMZN"

if len(stock_df) < 30:
    st.error("Not enough historical data to train a model.")
    st.stop()

# Prepare features
stock_df["lag_1"] = stock_df["close"].shift(1)
stock_df["lag_2"] = stock_df["close"].shift(2)
stock_df["lag_5"] = stock_df["close"].shift(5)
stock_df["ma_5"] = stock_df["close"].rolling(5).mean()
stock_df["ma_20"] = stock_df["close"].rolling(20).mean()

features = ["close", "lag_1", "lag_2", "lag_5", "ma_5", "ma_20"]

# Predict the next day's price change
stock_df["target_change"] = stock_df["close"].shift(-1) - stock_df["close"]

data = stock_df.dropna(subset=features + ["target_change"]).copy()

if len(data) < 30:
    st.error("Not enough usable rows after feature preparation.")
    st.stop()

X = data[features]
y = data["target_change"]

# Chronological train/test split
split = int(len(data) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=6,
    min_samples_leaf=10,
    max_features=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate on unseen historical data
predicted_change = model.predict(X_test)
predicted_prices = X_test["close"].to_numpy() + predicted_change
actual_prices = X_test["close"].to_numpy() + y_test.to_numpy()

mae = mean_absolute_error(actual_prices, predicted_prices)
rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))

baseline_prices = X_test["close"].to_numpy()
baseline_mae = mean_absolute_error(actual_prices, baseline_prices)

# Latest available prediction
latest_row = stock_df.dropna(subset=features).iloc[[-1]]
last_close = float(latest_row["close"].iloc[0])
predicted_change_next = float(model.predict(latest_row[features])[0])
future_price = last_close + predicted_change_next
last_date = latest_row["date"].iloc[0]

# Dashboard metrics
st.subheader(f"{stock} Stock Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Latest Closing Price", f"${last_close:.2f}")
col2.metric("Predicted Next Close", f"${future_price:.2f}",
            delta=f"${predicted_change_next:+.2f}")
col3.metric("Model MAE", f"${mae:.2f}")

col4, col5 = st.columns(2)
col4.metric("Model RMSE", f"${rmse:.2f}")
col5.metric("Baseline MAE", f"${baseline_mae:.2f}")

st.caption(
    f"Latest data date: {last_date:%Y-%m-%d}. "
    "The prediction is for the next observation after this date in the dataset, "
    "not necessarily the next calendar day or the current market day."
)

# Historical closing price chart
st.subheader("Historical Closing Prices")
history = stock_df.tail(250)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(history["date"], history["close"], label="Closing Price")
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.set_title(f"{stock} Historical Prices")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Actual vs predicted graph
st.subheader("Actual vs Predicted Prices")
fig2, ax2 = plt.subplots(figsize=(12, 4))
dates = data["date"].iloc[split:]

ax2.plot(dates, actual_prices, label="Actual Price")
ax2.plot(dates, predicted_prices, label="Predicted Price")
ax2.set_xlabel("Date")
ax2.set_ylabel("Price (USD)")
ax2.legend()
ax2.grid(alpha=0.3)
fig2.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

st.subheader("Model Evaluation")
if mae < baseline_mae:
    st.success("The model outperformed the simple baseline on MAE for this test period.")
else:
    st.warning("The simple baseline performed as well as or better than the model on MAE.")

st.write(
    "MAE measures average absolute prediction error. RMSE penalizes larger errors "
    "more heavily. Historical performance does not guarantee future accuracy."
)
