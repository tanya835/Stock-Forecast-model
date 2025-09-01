# stock.py
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import date
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# ------------------------
# Helper function: Theme
# ------------------------
def set_theme(bg_color="#f5f5f5", text_color="black"):
    st.markdown(
        f"""
        <style>
        body {{
            background-color: {bg_color};
            color: {text_color};
        }}
        .stTextInput label, .stPasswordInput label, .stSelectbox label {{
            color: {text_color} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ------------------------
# User Authentication Page
# ------------------------
def auth_page():
    set_theme(bg_color="#f5f5f5", text_color="black")
    st.title("🔐 User Authentication")

    if "users" not in st.session_state:
        st.session_state.users = {}
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "current_user" not in st.session_state:
        st.session_state.current_user = None
    if "page" not in st.session_state:
        st.session_state.page = "auth"

    option = st.radio("Choose option", ["Login", "Signup"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if option == "Signup":
        if st.button("Create Account"):
            if username in st.session_state.users:
                st.error("User already exists. Please login.")
            else:
                st.session_state.users[username] = password
                st.success("Account created successfully! Please login.")
    else:
        if st.button("Login"):
            if username in st.session_state.users and st.session_state.users[username] == password:
                st.session_state.authenticated = True
                st.session_state.current_user = username
                st.session_state.page = "forecast"
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")

# ------------------------
# Forecast Page
# ------------------------
def forecast_page():
    set_theme(bg_color="#5a5a5a", text_color="white")
    st.markdown(
        "<h2 style='text-align:center; color:white; background-color:grey; padding:14px; border-radius:8px;'>"
        "STOCK MARKET FORECASTING MODEL</h2>",
        unsafe_allow_html=True,
    )

    # Logout
    top = st.columns([1, 3, 1])
    with top[0]:
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.current_user = None
            st.session_state.page = "auth"
            st.rerun()

    # Company tickers
    name_to_ticker = {
        "Unitech Pvt Ltd": "UNITECH.NS",
        "Vodafone Idea": "IDEA.NS",
        "Reliance": "RELIANCE.NS",
        "Suzlon": "SUZLON.NS",
    }

    company = st.selectbox("Select a company", list(name_to_ticker.keys()))
    ticker = name_to_ticker[company]

    # Fetch data
    data = yf.download(ticker, period="5y", interval="1d")

    if data.empty:
        st.error("Could not fetch data. Try another ticker or check your internet.")
        return

    data = data.reset_index()

    # Plot actual stock prices
    st.subheader(f" Current Price — {company}")
    fig1, ax1 = plt.subplots()
    ax1.plot(data["Date"], data["Close"], label="Close")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price")
    ax1.legend()
    st.pyplot(fig1, clear_figure=True)

    # Prepare features
    X = np.array([d.toordinal() for d in data["Date"]]).reshape(-1, 1)
    y = data["Close"].values

    # Train models
    lr = LinearRegression().fit(X, y)
    rf = RandomForestRegressor(n_estimators=200, random_state=42).fit(X, y)

    # Prediction input
    st.subheader(" Predict stock price")
    max_date = date(2025, 12, 31)
    chosen_date = st.date_input(
        "Choose a date (till 31-12-2025)",
        min_value=data["Date"].min().date(),
        max_value=max_date,
        value=date(2025, 12, 25)
    )

    chosen_ord = np.array([[chosen_date.toordinal()]])

    # Predictions
    pred_lr = float(lr.predict(chosen_ord)[0])
    pred_rf = float(rf.predict(chosen_ord)[0])

    st.write(f"**Linear Regression prediction for {chosen_date}:** ₹{pred_lr:.2f}")
    st.write(f"**Random Forest prediction for {chosen_date}:** ₹{pred_rf:.2f}")

    # Forecast chart
    future_dates = pd.date_range(start=data["Date"].max(), end=max_date, freq="D")
    future_ord = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)

    preds_lr = lr.predict(future_ord)
    preds_rf = rf.predict(future_ord)

    fig2, ax2 = plt.subplots()
    ax2.plot(data["Date"], data["Close"], label="Actual")
    ax2.plot(future_dates, preds_lr, label="Linear Regression Forecast")
    ax2.plot(future_dates, preds_rf, label="Random Forest Forecast")
    ax2.axvline(chosen_date, color="white", linestyle="--", label=f"Chosen Date {chosen_date}")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Price")
    ax2.legend()
    st.pyplot(fig2, clear_figure=True)

# ------------------------
# App Routing
# ------------------------
if "page" not in st.session_state:
    st.session_state.page = "auth"

if st.session_state.page == "auth":
    auth_page()
elif st.session_state.page == "forecast":
    forecast_page()
