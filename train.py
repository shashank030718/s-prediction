# =====================================
# IMPORT LIBRARIES
# =====================================

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# =====================================
# DOWNLOAD STOCK DATA
# =====================================

ticker = "RELIANCE.NS"

print("Downloading stock data...\n")

df = yf.download(
    ticker,
    start="2015-01-01",
    end="2025-01-01"
)


# Fix MultiIndex columns from yfinance
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
# =====================================
# DISPLAY DATA
# =====================================

print("First 5 rows:\n")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

# =====================================
# PLOT CLOSING PRICE
# =====================================

plt.figure(figsize=(12,6))

plt.plot(df['Close'])

plt.title("Reliance Stock Closing Price")
plt.xlabel("Date")
plt.ylabel("Price")

plt.show()

# =====================================
# FEATURE ENGINEERING
# =====================================

print("\nCreating Moving Averages...\n")

# 10-day moving average
df['MA10'] = df['Close'].rolling(window=10).mean()

# 50-day moving average
df['MA50'] = df['Close'].rolling(window=50).mean()

# =====================================
# DISPLAY LAST 5 ROWS
# =====================================

print("\nLast 5 rows:\n")

print(
    df[
        ['Close', 'MA10', 'MA50']
    ].tail()
)



# Create target column
df['Target'] = df['Close'].shift(-1)

print(
    df[
        ['Close', 'MA10', 'MA50', 'Target']
    ].tail()
)




# Remove rows with missing values
df = df.dropna()

print("\nDataset after removing null values:")
print(df.shape)

# Features
X = df[['Close', 'MA10', 'MA50']]

# Target
y = df['Target']

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())
# =====================================
# PLOT MOVING AVERAGES
# =====================================

plt.figure(figsize=(15,8))

plt.plot(
    df['Close'],
    label='Close Price'
)

plt.plot(
    df['MA10'],
    label='MA10'
)

plt.plot(
    df['MA50'],
    label='MA50'
)

plt.title("Reliance Stock with Moving Averages")

plt.xlabel("Date")
plt.ylabel("Price")

plt.legend()

plt.show()


from sklearn.ensemble import RandomForestRegressor

# Create AI model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

print("\nTraining model...")

# Train model
model.fit(X, y)

print("Model trained successfully!")



# =====================================
# PREDICT NEXT DAY PRICE
# =====================================

latest_close = float(df['Close'].iloc[-1])
latest_ma10 = float(df['MA10'].iloc[-1])
latest_ma50 = float(df['MA50'].iloc[-1])

latest_data = pd.DataFrame({
    'Close': [latest_close],
    'MA10': [latest_ma10],
    'MA50': [latest_ma50]
})

prediction = model.predict(latest_data)

print("\n====================")
print(
    "Current Price:",
    round(latest_close, 2)
)

print(
    "Predicted Next Price:",
    round(float(prediction[0]), 2)
)
print("====================")


from sklearn.model_selection import train_test_split

model.fit(X, y)

# Split data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# Train model
model.fit(X_train, y_train)


test_predictions = model.predict(X_test)


from sklearn.metrics import (
    mean_absolute_error,
    r2_score
)


mae = mean_absolute_error(
    y_test,
    test_predictions
)

r2 = r2_score(
    y_test,
    test_predictions
)

print("\n====================")
print("MAE:", round(mae,2))
print("R2 Score:", round(r2,2))
print("====================")


import joblib

joblib.dump(
    model,
    "model/stock_model.pkl"
)