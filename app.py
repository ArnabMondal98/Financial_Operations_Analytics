# ===============================
# Financial Operations Analytics Dashboard
# Production Frontend Layout
# ===============================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Financial Operations Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/customers.csv")  # adjust path if needed
    df = pd.read_csv("data/transactions.csv")  # adjust path if needed
    st.write("Columns:", df.columns) 
    df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_data()

# -------------------------------
# SIDEBAR FILTERS
# -------------------------------
st.sidebar.title("📊 Dashboard Controls")

years = df["Date"].dt.year.unique()
selected_year = st.sidebar.selectbox("Select Year", sorted(years))

df = df[df["Date"].dt.year == selected_year]

if "Department" in df.columns:
    departments = st.sidebar.multiselect(
        "Select Department",
        df["Department"].unique(),
        default=df["Department"].unique()
    )
    df = df[df["Department"].isin(departments)]

# -------------------------------
# HEADER
# -------------------------------
st.title("📊 Financial Operations Analytics Dashboard")
st.markdown(
"""
Interactive dashboard for monitoring **financial performance**, 
**operational efficiency**, and **business KPIs**.
"""
)
st.divider()

# -------------------------------
# KPI CALCULATIONS
# -------------------------------
total_revenue = df["Revenue"].sum()
total_expense = df["Expense"].sum()
profit = total_revenue - total_expense
margin = (profit / total_revenue * 100) if total_revenue != 0 else 0

# -------------------------------
# KPI SECTION
# -------------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("💰 Total Revenue", f"{total_revenue:,.0f}")
col2.metric("💸 Total Expense", f"{total_expense:,.0f}")
col3.metric("📈 Profit", f"{profit:,.0f}")
col4.metric("📊 Profit Margin", f"{margin:.2f}%")

st.divider()

# -------------------------------
# REVENUE TREND
# -------------------------------
st.subheader("📈 Financial Trend Analysis")

trend = df.groupby("Date")[["Revenue", "Expense"]].sum().reset_index()

fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(trend["Date"], trend["Revenue"], label="Revenue")
ax1.plot(trend["Date"], trend["Expense"], label="Expense")
ax1.legend()
ax1.set_title("Revenue vs Expense Trend")

st.pyplot(fig1)

# -------------------------------
# DEPARTMENT ANALYSIS
# -------------------------------
if "Department" in df.columns:

    st.subheader("🏢 Department Expense Analysis")

    dept_expense = (
        df.groupby("Department")["Expense"]
        .sum()
        .sort_values(ascending=False)
    )

    fig2, ax2 = plt.subplots()
    sns.barplot(x=dept_expense.values, y=dept_expense.index, ax=ax2)
    ax2.set_title("Expense by Department")

    st.pyplot(fig2)

# -------------------------------
# PROFIT DISTRIBUTION
# -------------------------------
st.subheader("📊 Profit Distribution")

df["Profit"] = df["Revenue"] - df["Expense"]

fig3, ax3 = plt.subplots()
sns.histplot(df["Profit"], kde=True, ax=ax3)
ax3.set_title("Profit Distribution")

st.pyplot(fig3)

# -------------------------------
# DATA TABLE
# -------------------------------
st.subheader("🔎 Financial Data View")
st.dataframe(df, use_container_width=True)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption(
"Financial Operations Analytics Dashboard | Built with Streamlit & Python"
)


