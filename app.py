import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error
)
import requests

future_predictions = []

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="AI Stock Predictor",
    layout="wide"
)

# =====================================
# LOAD STOCK DATABASE
# =====================================

stocks = pd.read_csv(
    "nse_stocks.csv"
)

# Create Yahoo Finance ticker
stocks["Ticker"] = (
    stocks["SYMBOL"]
    + ".NS"
)

# =====================================
# TITLE
# =====================================

st.title("📈 AI Stock Price Predictor")

st.write(
    "Predict tomorrow's stock price using Machine Learning"
)

# =====================================
# SEARCH COMPANY USING YAHOO
# =====================================


search = st.text_input(
    "🔍 Search Company or Symbol"
)

ticker = None
selected_company = None
selected_exchange = "NSE"

if search:

    matches = stocks[

        stocks[
            "NAME OF COMPANY"
        ]
        .str.contains(
            search,
            case=False,
            na=False
        )

        |

        stocks[
            "SYMBOL"
        ]
        .str.contains(
            search,
            case=False,
            na=False
        )
    ]

    if not matches.empty:

        matches = matches.copy()

        # Display format
        matches["Display"] = (

            matches[
                "NAME OF COMPANY"
            ]

            + " | "

            + matches[
                "SYMBOL"
            ]

            + " | "

            + matches[
                "Ticker"
            ]
        )

        selected = st.selectbox(
            "Select Stock",
            matches[
                "Display"
            ]
        )

        row = matches[
            matches[
                "Display"
            ]
            == selected
        ].iloc[0]

        ticker = row[
            "Ticker"
        ]

        selected_company = row[
            "NAME OF COMPANY"
        ]

        st.success(
            f"Selected: {ticker}"
        )

        # Sidebar company info
        st.sidebar.header(
            "Company Information"
        )

        st.sidebar.write(
            "Company:",
            row[
                "NAME OF COMPANY"
            ]
        )

        st.sidebar.write(
            "Symbol:",
            row[
                "SYMBOL"
            ]
        )

        st.sidebar.write(
            "Ticker:",
            ticker
        )

        st.sidebar.write(
            "Listed:",
            row[
                " DATE OF LISTING"
            ]
        )

        st.sidebar.write(
            "ISIN:",
            row[
                " ISIN NUMBER"
            ]
        )

        st.sidebar.write(
            "Face Value:",
            row[
                " FACE VALUE"
            ]
        )

    else:

        st.error(
            "No stock found"
        )

# =====================================
# PREDICT
# =====================================

if ticker and st.button("Predict"):

    with st.spinner(
        "Downloading data and training AI..."
    ):

        # Download historical data
        df = yf.download(
            ticker,
            start="2015-01-01",
            progress=False
        )

        if df.empty:
            st.error(
                "Unable to download stock data."
            )
            st.stop()

        # Fix multi-index
        if isinstance(
            df.columns,
            pd.MultiIndex
        ):
            df.columns = (
                df.columns
                .get_level_values(0)
            )

        # Feature Engineering
        df['MA10'] = (
            df['Close']
            .rolling(10)
            .mean()
        )

        df['MA50'] = (
            df['Close']
            .rolling(50)
            .mean()
        )

        # Target
        df['Target'] = (
            df['Close']
            .shift(-1)
        )

        df = df.dropna()

        # Features & Target
        X = df[
            [
                'Close',
                'MA10',
                'MA50'
            ]
        ]

        y = df['Target']

        # Train/Test Split
        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )
        )

        # Model
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

        model.fit(
            X_train,
            y_train
        )

        # Accuracy
        y_pred = model.predict(
            X_test
        )

        r2 = r2_score(
            y_test,
            y_pred
        )

        mae = mean_absolute_error(
            y_test,
            y_pred
        )

        # Live price
        stock = yf.Ticker(
            ticker
        )

        try:
            current_price = float(
                stock.fast_info[
                    'lastPrice'
                ]
            )

        except:
            current_price = float(
                df['Close']
                .iloc[-1]
            )

        previous_close = float(
            df['Close']
            .iloc[-1]
        )

        # Tomorrow prediction
        latest = pd.DataFrame({

            'Close': [
                current_price
            ],

            'MA10': [
                float(
                    df['MA10']
                    .iloc[-1]
                )
            ],

            'MA50': [
                float(
                    df['MA50']
                    .iloc[-1]
                )
            ]
        })

        prediction = model.predict(
            latest
        )

        predicted_price = float(
            prediction[0]
        )

        # Next 7 days prediction
        future_predictions = []

        close = current_price

        ma10 = float(
            df['MA10']
            .iloc[-1]
        )

        ma50 = float(
            df['MA50']
            .iloc[-1]
        )

        for i in range(7):

            future_input = pd.DataFrame({

                'Close': [close],

                'MA10': [ma10],

                'MA50': [ma50]
            })

            next_price = float(
                model.predict(
                    future_input
                )[0]
            )

            future_predictions.append(
                next_price
            )

            close = next_price

        # Change
        change = (
            predicted_price
            - current_price
        )

        change_percent = (
            change
            / current_price
        ) * 100

    # =====================
    # DASHBOARD
    # =====================

    st.header(
        selected_company
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Current Price",
        f"₹{current_price:.2f}"
    )

    col2.metric(
        "Tomorrow",
        f"₹{predicted_price:.2f}"
    )

    col3.metric(
        "Change",
        f"{change_percent:.2f}%"
    )

    col4, col5, col6, col7 = st.columns(4)

    col4.metric(
        "Previous Close",
        f"₹{previous_close:.2f}"
    )

    col5.metric(
        "Exchange",
        selected_exchange
    )

    col6.metric(
        "R² Score",
        f"{r2:.2f}"
    )

    col7.metric(
        "MAE",
        f"₹{mae:.2f}"
    )

    # Buy/Sell
    if predicted_price > current_price:

        st.success(
            "📈 BUY SIGNAL"
        )

    else:

        st.error(
            "📉 SELL SIGNAL"
        )

    # Historical Chart
    st.subheader(
        "📊 Historical Chart"
    )

    fig = plt.figure(
        figsize=(15,7)
    )

    plt.plot(
        df['Close'],
        label='Close'
    )

    plt.plot(
        df['MA10'],
        label='MA10'
    )

    plt.plot(
        df['MA50'],
        label='MA50'
    )

    plt.legend()

    st.pyplot(fig)

    # Forecast Table
    st.subheader(
        "📅 Next 7-Day Prediction"
    )

    forecast = pd.DataFrame({

        "Day": [
            "Day 1",
            "Day 2",
            "Day 3",
            "Day 4",
            "Day 5",
            "Day 6",
            "Day 7"
        ],

        "Predicted Price":
            future_predictions
    })

    st.dataframe(
        forecast,
        use_container_width=True
    )

    # Forecast Chart
    st.subheader(
        "📈 Next 7-Day Forecast"
    )

    fig2 = plt.figure(
        figsize=(10,5)
    )

    plt.plot(
        forecast[
            "Predicted Price"
        ],
        marker='o'
    )

    plt.xticks(
        range(7),
        forecast[
            "Day"
        ]
    )

    plt.title(
        "Next 7-Day Forecast"
    )

    plt.ylabel(
        "Price"
    )

    st.pyplot(fig2)