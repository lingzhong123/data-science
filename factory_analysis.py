import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- 1. Page Config ---
st.set_page_config(page_title="Case analysis", layout="wide")

# --- 2. Data Loading ---
@st.cache_data
def load_data():
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
    fixed_cols = ["Labor Cost - Engineer and Management", "Rent", "Janitor", "Machine Maintainance Cost", "Insurance", "Depreciation"]
    var_cols = ["Labor Cost - Operator", "Travel", "Company Transportation", "Utilities Electricity", "Material Cost", "Freight"]
    
    df['Fixed_Cost'] = df[fixed_cols].sum(axis=1)
    df['Variable_Cost'] = df[var_cols].sum(axis=1)
    df['Total_Cost'] = df['Fixed_Cost'] + df['Variable_Cost']
    df['CPU_USD'] = df['Total_Cost'] / df['Total Sales Qty']
    return df, fixed_cols, var_cols

df, fixed_items, variable_items = load_data()

st.title("📊 Cost Report (2024-2026)")
st.divider()

# --- MODULE 1: COST CLASSIFICATION ---
st.header("1. Cost Classification Analysis")
c1, c2 = st.columns(2)
with c1:
    st.subheader("📌 Fixed Cost Items")
    st.info("\n".join([f"- {i}" for i in fixed_items]))
with c2:
    st.subheader("⚡ Variable Cost Items")
    st.warning("\n".join([f"- {i}" for i in variable_items]))

st.subheader("Cost vs. Production Volume Trends")
# Normalized Trend Line
df_norm = df[['Total_Cost', 'Total Sales Qty']].apply(lambda x: x / x.iloc[0] * 100)
fig_trend = px.line(df_norm, title="Relative Growth: Total Cost vs. Sales Qty (Baseline = 100)")
fig_trend.add_vline(x=8, line_dash="dash", line_color="red") # Q1'26 split
st.plotly_chart(fig_trend, use_container_width=True)

# --- MODULE 2: COST SAVING OPPORTUNITIES ---
st.header("2. Cost Saving Opportunities")

# A. Pie Chart Comparison
st.subheader("A. Structural Shift: Actual (Avg) vs. Forecast (Q1'26)")
col_a, col_b = st.columns(2)
actual_avg = df.loc["Q1'24":"Q4'25"].mean()
with col_a:
    st.plotly_chart(px.pie(values=[actual_avg['Fixed_Cost'], actual_avg['Variable_Cost']], names=['Fixed', 'Variable'], title="Historical Avg Structure", hole=0.4), use_container_width=True)
with col_b:
    st.plotly_chart(px.pie(values=[df.loc["Q1'26", 'Fixed_Cost'], df.loc["Q1'26", 'Variable_Cost']], names=['Fixed', 'Variable'], title="Q1'26 Forecast Structure", hole=0.4), use_container_width=True)

# B. Cost Drivers (Waterfall/Bar)
st.subheader("B. Cost Growth Contribution (Q4'25 to Q1'26)")
q4_25 = df.loc["Q4'25", variable_items + fixed_items]
q1_26 = df.loc["Q1'26", variable_items + fixed_items]
diff = (q1_26 - q4_25).sort_values(ascending=False)
fig_diff = px.bar(diff, title="Absolute USD Increase per Element", color=diff.values, labels={'value':'USD Increase', 'index':'Category'})
st.plotly_chart(fig_diff, use_container_width=True)

# C. Exchange Rate vs CPU
st.subheader("C. FX Exposure & Unit Cost Trend")
fig_fx = go.Figure()
fig_fx.add_trace(go.Scatter(x=df.index, y=df['Exchange Rate'], name="Exchange Rate (USD/MYR)", yaxis="y1"))
fig_fx.add_trace(go.Scatter(x=df.index, y=df['CPU_USD'], name="Unit Cost (USD)", yaxis="y2"))
fig_fx.update_layout(yaxis=dict(title="FX Rate"), yaxis2=dict(title="CPU USD", overlaying='y', side='right'))
st.plotly_chart(fig_fx, use_container_width=True)

st.success("**Summary of Saving Opportunities:** Focus on **Labor Efficiency** (biggest grower) and **Utility Consumption** during peak quarters. Manage **FX exposure** as the MYR strengthens, impacting local purchasing power.")

# --- MODULE 3: PROPOSAL OR RECOMMENDATION ---
st.header("3. Strategic Recommendations & Challenges")

recommendations = [
    {"Category": "Labor Cost - Operator", "Red Flags": "High Q1'26 forecast despite new equipment; similar Q3/Q4 patterns in past.", "Proposed Action": "Check for overestimated OT or underestimated efficiency gains from new equipment."},
    {"Category": "Total Sales Qty", "Red Flags": "Q1'26 growth deviates from historical seasonality.", "Proposed Action": "Provide business drivers for Q1 peak (e.g., new orders) or align with history."},
    {"Category": "Travel & Transport", "Red Flags": "Fixed annual values ($15/$356) suggest lack of activity-based planning.", "Proposed Action": "Provide a 2026 travel plan or activity-based forecast for justification."},
    {"Category": "Maintenance", "Red Flags": "Static at $250 for 3 years despite rising depreciation & output.", "Proposed Action": "Verify why costs are static while asset usage and production volume increase."}
]

st.table(pd.DataFrame(recommendations))
