import streamlit as st
import pandas as pd

st.title("Financial Operations Analytics Dashboard")

df = pd.read_csv("outputs/unified/dashboard/fact_customer_metrics.csv")

st.write("Customer Metrics Sample")
st.dataframe(df.head())

st.metric("Total Customers", df['customer_id'].nunique())
st.metric("Total Revenue", round(df['total_revenue'].sum(),2))
