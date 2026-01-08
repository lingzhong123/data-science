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
    
    return df, fixed_items, variable_items

df, fixed_items, variable_items = load_and_process_data()

# --- 3. Header ---
st.title("🏭 Factory Financial Cost Audit Dashboard")
st.markdown("### Strategic Cost Management (2024 - 2026)")
st.divider()

# --- 4. Cost Classification Panel ---
st.header("1. Cost Classification Analysis")
col_f, col_v = st.columns(2)

with col_f:
    st.subheader("📌 Fixed Cost Items")
    st.info("\n".join([f"- {item}" for item in fixed_items]))

with col_v:
    st.subheader("⚡ Variable Cost Items")
    st.warning("\n".join([f"- {item}" for item in variable_items]))

# --- 5. 3-Year Cost Trend Line Chart (FIXED ERROR) ---
st.header("2. 3-Year Cost Evolution (Actual vs. Forecast)")

trend_fig = go.Figure()
trend_fig.add_trace(go.Scatter(x=df.index, y=df['Total_Cost'], name='Total Cost', line=dict(color='black', width=3)))
trend_fig.add_trace(go.Scatter(x=df.index, y=df['Variable_Cost'], name='Variable Cost', line=dict(color='#F39C12', width=2)))
trend_fig.add_trace(go.Scatter(x=df.index, y=df['Fixed_Cost'], name='Fixed Cost', line=dict(color='#2E86C1', width=2)))

# Get the numeric index of Q1'26 to place the vertical line safely
try:
    forecast_idx = list(df.index).index("Q1'26")
    trend_fig.add_vline(x=forecast_idx, line_dash="dash", line_color="red")
    # Add annotation separately to avoid the mean() calculation error
    trend_fig.add_annotation(x=forecast_idx, y=df['Total_Cost'].max(),
                text="Forecast Starts", showarrow=False, xanchor="left", font=dict(color="red"))
except ValueError:
    pass

trend_fig.update_layout(
    title="Cost Trends: Historical Actuals vs. 2026 Projections",
    xaxis_title="Quarter",
    yaxis_title="Amount (USD)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(trend_fig, use_container_width=True)

# --- 6. Structure Comparison ---
st.header("3. Cost Mix Comparison")
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

# --- 7. Efficiency Audit Panel ---
st.header("4. Audit & Performance Inquiries")

# Efficiency calculations
df['Qty_Change'] = df['Total Sales Qty'].pct_change()
df['VC_Change'] = df['Variable_Cost'].pct_change()
df['Elasticity'] = df['VC_Change'] / df['Qty_Change']

bad_planning = df[df['Elasticity'] > 1.15][['Total Sales Qty', 'Variable_Cost', 'Elasticity']]

st.subheader("Efficiency Tracker (Variable Cost Elasticity)")
st.write("Quarters where Variable Costs scale significantly faster than Production Volume:")

try:
    st.dataframe(bad_planning.style.background_gradient(cmap='Reds', subset=['Elasticity']).format("{:.2f}", subset=['Elasticity']))
except:
    st.dataframe(bad_planning)

with st.expander("Strategic Audit Questions"):
    # Reference value safely
    q1_26_val = df.loc["Q1'26", "Elasticity"]
    st.markdown(f"""
    - **Operational Efficiency:** The elasticity for Q1'26 is **{q1_26_val:.2f}**. This indicates a loss of economies of scale. 
    - **Fixed Overhead:** Fixed costs are rising despite no significant increase in base capacity. Justify the $1,700 Depreciation forecast.
    """)
