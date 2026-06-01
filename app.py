import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# -----------------------
# PAGE CONFIG
# -----------------------

st.set_page_config(
    page_title="AI Demand Forecast Dashboard",
    page_icon="📈",
    layout="wide"
)

# -----------------------
# LOAD DATA
# -----------------------

@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_data.csv")
    return df

@st.cache_resource
def load_model():
    model = joblib.load("demand_forecast_model.pkl")
    return model

df = load_data()
model = load_model()

# -----------------------
# HEADER
# -----------------------

st.title("📈 AI-Powered Retail Demand Forecasting Dashboard")

st.markdown("""
Analyze demand trends, demand distribution,
inventory insights, and predict future demand
using Machine Learning.
""")

st.divider()

# ---------------------------
# SIDEBAR FILTERS
# ---------------------------

st.sidebar.header("🔍 Filters")

selected_region = st.sidebar.multiselect(
    "Region",
    options=df["Region"].unique(),
    default=df["Region"].unique()
)

selected_category = st.sidebar.multiselect(
    "Category",
    options=df["Category"].unique(),
    default=df["Category"].unique()
)

filtered_df = df[
    (df["Region"].isin(selected_region)) &
    (df["Category"].isin(selected_category))
]

# ---------------------------
# KPI SECTION
# ---------------------------

total_demand = filtered_df["Demand Forecast"].sum()
avg_demand = filtered_df["Demand Forecast"].mean()
inventory = filtered_df["Inventory Level"].sum()
products = filtered_df["Product ID"].nunique()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Demand",
        f"{total_demand:,.0f}"
    )

with col2:
    st.metric(
        "Average Demand",
        f"{avg_demand:.2f}"
    )

with col3:
    st.metric(
        "Inventory Level",
        f"{inventory:,.0f}"
    )

with col4:
    st.metric(
        "Products",
        products
    )

st.divider()

# ---------------------------
# DEMAND DISTRIBUTION
# ---------------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("Demand Distribution")
    
    fig = px.histogram(
        filtered_df,
        x="Demand Forecast",
        nbins=30,
        title="Demand Distribution"
    )
    
    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:
    st.subheader("Demand by Category")
    
    category_demand = (
        filtered_df
        .groupby("Category")["Demand Forecast"]
        .mean()
        .reset_index()
    )
    
    fig = px.bar(
        category_demand,
        x="Category",
        y="Demand Forecast",
        title="Average Demand by Category"
    )
    
    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.divider()

# ---------------------------
# MONTHLY TREND
# ---------------------------

st.subheader("Monthly Demand Trend")

monthly = (
    filtered_df
    .groupby("Month")["Demand Forecast"]
    .mean()
    .reset_index()
)

fig = px.line(
    monthly,
    x="Month",
    y="Demand Forecast",
    markers=True,
    title="Monthly Demand Trend"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

# ---------------------------
# REGION ANALYSIS
# ---------------------------

st.subheader("Region Wise Demand")

region = (
    filtered_df
    .groupby("Region")["Demand Forecast"]
    .mean()
    .reset_index()
)

fig = px.bar(
    region,
    x="Region",
    y="Demand Forecast",
    title="Region Wise Demand"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

# ---------------------------
# TOP PRODUCTS
# ---------------------------

st.subheader("Top Products")

top_products = (
    filtered_df
    .groupby("Product ID")["Demand Forecast"]
    .sum()
    .reset_index()
    .sort_values(
        by="Demand Forecast",
        ascending=False
    )
    .head(10)
)

st.dataframe(
    top_products,
    use_container_width=True
)

st.divider()

# ---------------------------
# AI DEMAND PREDICTOR
# ---------------------------

st.subheader("🤖 AI Demand Predictor")

st.info(
    "Enter business parameters to estimate demand."
)

col1, col2, col3 = st.columns(3)

with col1:
    inventory_level = st.number_input(
        "Inventory Level",
        value=500
    )
    units_ordered = st.number_input(
        "Units Ordered",
        value=200
    )

with col2:
    price = st.number_input(
        "Price",
        value=50.0
    )
    discount = st.number_input(
        "Discount",
        value=10.0
    )

with col3:
    competitor_price = st.number_input(
        "Competitor Pricing",
        value=55.0
    )

if st.button("Predict Demand"):
    if len(filtered_df) == 0:
        st.error("No data available to estimate base features. Please adjust filters in the sidebar.")
    else:
        sample = filtered_df.iloc[0].copy()
        
        sample["Inventory Level"] = inventory_level
        sample["Units Ordered"] = units_ordered
        sample["Price"] = price
        sample["Discount"] = discount
        sample["Competitor Pricing"] = competitor_price
        
        prediction_input = pd.DataFrame(
            [sample]
        )
        
        prediction_input = prediction_input.drop(
            columns=[
                "Demand Forecast"
            ],
            errors="ignore"
        )
        
        prediction = model.predict(
            prediction_input
        )[0]
        
        st.success(
            f"Predicted Demand: {prediction:.2f} Units"
        )

st.divider()

st.caption(
    "Built using Streamlit, XGBoost and Machine Learning"
)
