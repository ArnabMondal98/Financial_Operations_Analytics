# =====================================
# Financial Operations Analytics Dashboard
# Production Version (Deployment Safe)
# =====================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------
# PAGE CONFIG
# -------------------------------------
st.set_page_config(
    page_title="Financial Operations Analytics",
    page_icon="📊",
    layout="wide"
)

# -------------------------------------
# LOAD DATA
# -------------------------------------
@st.cache_data
def load_data():

    df = pd.read_csv("data/transactions.csv")

    # Clean column names
    df.columns = df.columns.str.strip().str.lower()

    # Convert date column
    df["date"] = pd.to_datetime(df["transaction_date"])

    # Business metrics mapping
    df["revenue"] = df["amount"]
    df["expense"] = 0  # no expense data available
    df["profit"] = df["revenue"] - df["expense"]

    return df


df = load_data()

# -------------------------------------
# SIDEBAR FILTER
# -------------------------------------
st.sidebar.title("📊 Filters")

years = sorted(df["date"].dt.year.unique())
selected_year = st.sidebar.selectbox("Select Year", years)

df = df[df["date"].dt.year == selected_year]

# -------------------------------------
# HEADER
# -------------------------------------
st.title("📊 Financial Operations Analytics Dashboard")
st.caption("Interactive financial performance monitoring")

st.divider()

# -------------------------------------
# KPI SECTION
# -------------------------------------
total_revenue = df["revenue"].sum()
total_expense = df["expense"].sum()
profit = df["profit"].sum()

col1, col2, col3 = st.columns(3)

col1.metric("💰 Total Revenue", f"{total_revenue:,.0f}")
col2.metric("💸 Total Expense", f"{total_expense:,.0f}")
col3.metric("📈 Profit", f"{profit:,.0f}")

st.divider()

# -------------------------------------
# REVENUE TREND
# -------------------------------------
st.subheader("📈 Revenue Trend")

trend = (
    df.groupby("date")["revenue"]
    .sum()
    .reset_index()
)

fig, ax = plt.subplots()
ax.plot(trend["date"], trend["revenue"])
ax.set_title("Revenue Over Time")
ax.set_xlabel("Date")
ax.set_ylabel("Revenue")

st.pyplot(fig)

# -------------------------------------
# PAYMENT METHOD ANALYSIS
# -------------------------------------
st.subheader("💳 Revenue by Payment Method")

payment_summary = (
    df.groupby("payment_method")["revenue"]
    .sum()
    .sort_values(ascending=False)
)

st.bar_chart(payment_summary)

# -------------------------------------
# DATA VIEW
# -------------------------------------
st.subheader("🔎 Transaction Data")
st.dataframe(df, use_container_width=True)

st.markdown("---")
st.caption("Built with Streamlit | Financial Operations Analytics")
