import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- 1. Page Configuration ---
st.set_page_config(page_title="Factory Cost Audit Dashboard", layout="wide")

# --- 2. Data Processing ---
@st.cache_data
def load_and_process_data():
    data = {
        "Metric": ["Labor Cost - Engineer and Management", "Labor Cost - Operator", "Travel", "Company Transportation", "Rent", "Utilities Electricity", "Janitor", "Material Cost", "Machine Maintainance Cost", "Freight", "Insurance", "Depreciation", "Total Sales Qty", "Exchange Rate"],
        "Q1'24": [100, 2000, 20, 200, 40, 2500, 50, 750, 250, 125, 50, 1000, 25000, 4.78],
        "Q2'24": [101, 2230, 30, 203, 41, 2737, 51, 825, 250, 138, 50, 990, 27500, 4.71],
        "Q3'24": [109, 2042, 10, 330, 44, 2588, 55, 701, 250, 117, 50, 980, 23375, 4.37],
        "Q4'24": [107, 2197, 11, 323, 43, 2731, 53, 771, 250, 129, 50, 1000, 25713, 4.47],
        "Q1'25": [115, 2381, 19, 327, 43, 2786, 54, 779, 250, 130, 55, 1200, 25970, 4.42],
        "Q2'25": [120, 2734, 29, 341, 45, 3144, 56, 857, 250, 143, 55, 1400, 28567, 4.23],
        "Q3'25": [120, 2328, 10, 342, 45, 2762, 57, 728, 250, 121, 55, 1400, 24282, 4.22],
        "Q4'25": [122, 2617, 10, 349, 46, 3047, 58, 801, 250, 134, 55, 1500, 26710, 4.13],
        "Q1'26": [132, 3960, 15, 356, 47, 4113, 59, 1122, 250, 187, 55, 1600, 37394, 4.05],
        "Q2'26": [132, 2772, 15, 356, 47, 3056, 59, 785, 250, 131, 55, 1600, 26176, 4.05],
        "Q3'26": [132, 3784, 15, 356, 47, 3179, 59, 825, 250, 137, 55, 1700, 27484, 4.05],
        "Q4'26": [132, 4540, 15, 356, 47, 3697, 59, 989, 250, 165, 55, 1700, 32981, 4.05],
    }
    df = pd.DataFrame(data).set_index("Metric").T
    
    fixed_items = ["Labor Cost - Engineer and Management", "Rent", "Janitor", "Machine Maintainance Cost", "Insurance", "Depreciation"]
    variable_items = ["Labor Cost - Operator", "Travel", "Company Transportation", "Utilities Electricity", "Material Cost", "Freight"]
    
    df['Fixed_Cost'] = df[fixed_items].sum(axis=1)
    df['Variable_Cost'] = df[variable_items].sum(axis=1)
    df['Total_Cost'] = df['Fixed_Cost'] + df['Variable_Cost']
    df['CPU_USD'] = df['Total_Cost'] / df['Total Sales Qty']
    df['Elasticity'] = df['Variable_Cost'].pct_change() / df['Total Sales Qty'].pct_change()
    return df

df = load_and_process_data()

# --- 3. Header ---
st.title("🏭 Factory Financial Audit Dashboard")
st.info("Goal: Contrast Historical Reality vs. Forecasted Efficiency.")

# --- 4. Section 1: Structure Comparison ---
st.header("1. Cost Mix: Historical Avg (2024-25) vs. Forecast (Q1'26)")
col1, col2 = st.columns(2)

actual_avg = df.loc["Q1'24":"Q4'25"].mean()
forecast_q126 = df.loc["Q1'26"]

with col1:
    fig_actual = go.Figure(data=[go.Pie(
        labels=['Fixed Costs', 'Variable Costs'],
        values=[actual_avg['Fixed_Cost'], actual_avg['Variable_Cost']],
        hole=.4, title="Historical Avg (24-25)",
        marker_colors=['#2E86C1', '#F39C12']
    )])
    st.plotly_chart(fig_actual, use_container_width=True)

with col2:
    fig_forecast = go.Figure(data=[go.Pie(
        labels=['Fixed Costs', 'Variable Costs'],
        values=[forecast_q126['Fixed_Cost'], forecast_q126['Variable_Cost']],
        hole=.4, title="Forecast Q1'26",
        marker_colors=['#1B4F72', '#D68910']
    )])
    st.plotly_chart(fig_forecast, use_container_width=True)

# --- 5. Section 2: Efficiency Audit ---
st.header("2. Saving Opportunities & Efficiency Audit")
col3, col4 = st.columns(2)

with col3:
    st.subheader("Utility Consumption Audit")
    fig_leak = px.scatter(df, x="Total Sales Qty", y="Utilities Electricity", 
                          trendline="ols", size="CPU_USD", color="CPU_USD",
                          title="Electricity vs Production Trend")
    st.plotly_chart(fig_leak, use_container_width=True)

with col4:
    st.subheader("Labor Cost per Unit (CPU)")
    df['Operator_CPU'] = df['Labor Cost - Operator'] / df['Total Sales Qty']
    fig_roi = px.line(df, y='Operator_CPU', markers=True, title="Unit Labor Cost Trend (USD)")
    st.plotly_chart(fig_roi, use_container_width=True)

# --- 6. Section 3: Audit & Challenges ---
st.header("3. Executive Audit Panel")
bad_planning = df[df['Elasticity'] > 1.15][['Total Sales Qty', 'Variable_Cost', 'Elasticity']]

try:
    st.dataframe(
        bad_planning.style.background_gradient(cmap='Reds', subset=['Elasticity'])
        .format("{:.2f}", subset=['Elasticity'])
    )
except Exception:
    st.dataframe(bad_planning)

st.subheader("Strategic Challenges:")
with st.expander("Specific Questions for Cost Owners"):
    # Corrected quotes for Q1'26 index
    elasticity_val = df.loc["Q1'26", "Elasticity"]
    st.markdown(f"""
    - **Production Manager (Q1'26 Forecast):** The Variable Cost Elasticity is **{elasticity_val:.2f}**. Why are costs scaling faster than production volume?
    - **Engineering Dept:** Points significantly above the regression line in the Utility Audit indicate wasted energy or poor machine scheduling.
    - **Finance (Automation ROI):** Despite higher depreciation, Unit Labor Cost (CPU) is not showing significant reduction in 2026.
    """)

# --- Sidebar Survival Check ---
st.sidebar.header("Survival Test (MYR)")
mock_fx = st.sidebar.slider("Simulated USD/MYR", 3.80, 5.00, 4.05)
# FIXED SYNTAX: Double quotes for the index containing a single quote
q1_26_myr = df.loc["Q1'26", "CPU_USD"] * mock_fx
st.sidebar.metric("Simulated Unit Cost (MYR)", f"RM {q1_26_myr:.3f}")
