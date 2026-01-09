import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- 1. Page Config ---
st.set_page_config(page_title="Case Analysis Chunling Zhong", layout="wide")

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
    df_raw = pd.DataFrame(data).set_index("Metric")
    df = df_raw.T
    
    fixed_cols = ["Labor Cost - Engineer and Management", "Rent", "Janitor", "Company Transportation", "Machine Maintainance Cost", "Insurance", "Depreciation"]
    var_cols = ["Labor Cost - Operator", "Travel", "Utilities Electricity", "Material Cost", "Freight"]
    
    df['Fixed_Cost'] = df[fixed_cols].sum(axis=1)
    df['Variable_Cost'] = df[var_cols].sum(axis=1)
    df['Total_Cost'] = df['Fixed_Cost'] + df['Variable_Cost']
    df['CPU_USD'] = df['Total_Cost'] / df['Total Sales Qty']
    return df, df_raw, fixed_cols, var_cols

df, df_raw, fixed_items, variable_items = load_data()

st.title("📊 Case Analysis")

with st.expander("🔍 View Raw Financial Data"):
    st.dataframe(df_raw.style.format("{:,.0f}"))

st.divider()

# --- MODULE 1: COST CLASSIFICATION & TRENDS ---
st.header("1. Cost Classification & Growth Trends")
c1, c2 = st.columns(2)
with c1:
    st.info("**Fixed Costs:**\n" + "\n".join([f"- {i}" for i in fixed_items]))
with c2:
    st.warning("**Variable Costs:**\n" + "\n".join([f"- {i}" for i in variable_items]))

# Macro Index: Total Cost vs Sales
df_norm = df[['Total_Cost', 'Total Sales Qty']].apply(lambda x: x / x.iloc[0] * 100)
st.plotly_chart(px.line(df_norm, title="Macro Growth Index: Total Cost vs. Sales Qty (Q1'24 = 100)"), use_container_width=True)

# NEW: Detailed Item Trends vs Sales Volume
st.subheader("Detailed Item Trends vs. Sales Volume (Base Q1'24 = 100)")
df_all_indices = df_raw.T.apply(lambda x: x / x.iloc[0] * 100)

fig_all = go.Figure()

# Sales Volume Reference Line (Bold/Dashed)
fig_all.add_trace(go.Scatter(
    x=df_all_indices.index, y=df_all_indices["Total Sales Qty"],
    name="Total Sales Qty (Baseline)",
    line=dict(color='black', width=4, dash='dash'),
    mode='lines+markers'
))

# All other cost items
for col in df_all_indices.columns:
    if col not in ["Total Sales Qty", "Exchange Rate"]:
        fig_all.add_trace(go.Scatter(
            x=df_all_indices.index, y=df_all_indices[col],
            name=col, mode='lines', line=dict(width=1.5),
            visible='legendonly' # User can click to toggle specific items
        ))

fig_all.update_layout(
    title="Cost Item Variance vs. Volume Growth",
    hovermode="x unified",
    xaxis_title="Quarter",
    yaxis_title="Index (Q1'24 = 100)",
    height=500,
    legend=dict(orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=0.5)
)
st.plotly_chart(fig_all, use_container_width=True)
st.caption("Audit Tip: Use the legend to toggle specific costs. Lines rising significantly above the black dashed line indicate potential efficiency loss.")

# --- MODULE 2: COST SAVING OPPORTUNITIES ---
st.header("2. Cost Saving Opportunities")

st.subheader("A. Variance Analysis: Q4'25 vs. Q1'26 Spending Bridge")
q4_vals = df.loc["Q4'25", fixed_items + variable_items]
q1_vals = df.loc["Q1'26", fixed_items + variable_items]
bridge_diff = (q1_vals - q4_vals).to_dict()

x_labels = ["Q4'25 Total"] + list(bridge_diff.keys()) + ["Q1'26 Total"]
measure = ["absolute"] + ["relative"] * len(bridge_diff) + ["total"]
y_values = [df.loc["Q4'25", 'Total_Cost']] + list(bridge_diff.values()) + [0]

fig_bridge = go.Figure(go.Waterfall(
    name="Spending Bridge", orientation="v",
    measure=measure, x=x_labels, y=y_values,
    connector={"line":{"color":"rgb(63, 63, 63)"}},
    increasing={"marker":{"color":"#EF553B"}}, 
    decreasing={"marker":{"color":"#00CC96"}}, 
    totals={"marker":{"color":"#636EFA"}}
))
fig_bridge.update_layout(title="Total Spending Bridge (USD Variance)", showlegend=False)
st.plotly_chart(fig_bridge, use_container_width=True)

col_pie1, col_pie2 = st.columns(2)
actual_avg = df.loc["Q1'24":"Q4'25"].mean()
with col_pie1:
    st.plotly_chart(px.pie(values=[actual_avg['Fixed_Cost'], actual_avg['Variable_Cost']], names=['Fixed', 'Variable'], title="Historical Avg Structure", hole=0.4), use_container_width=True)
with col_pie2:
    st.plotly_chart(px.pie(values=[df.loc["Q1'26", 'Fixed_Cost'], df.loc["Q1'26", 'Variable_Cost']], names=['Fixed', 'Variable'], title="Q1'26 Forecast Structure", hole=0.4), use_container_width=True)

st.subheader("B. FX Exposure & Unit Cost Trend")
fig_fx = go.Figure()
fig_fx.add_trace(go.Scatter(x=df.index, y=df['Exchange Rate'], name="USD/MYR Rate", yaxis="y1"))
fig_fx.add_trace(go.Scatter(x=df.index, y=df['CPU_USD'], name="CPU (USD)", yaxis="y2"))
fig_fx.update_layout(yaxis=dict(title="Exchange Rate"), yaxis2=dict(title="Unit Cost", overlaying='y', side='right'))
st.plotly_chart(fig_fx, use_container_width=True)

# --- MODULE 3: STRATEGIC RECOMMENDATIONS ---
st.header("3. Strategic Recommendations & Challenges")

st.subheader("Visual Efficiency of Scale in 2026")
ev_col1, ev_col2 = st.columns(2)

quarters_26 = ["Q1'26", "Q2'26", "Q3'26", "Q4'26"]
df_26 = df.loc[quarters_26].copy()
df_26['Quarter'] = df_26.index

with ev_col1:
    fig_quad = px.scatter(df_26, x="Total Sales Qty", y="CPU_USD", text="Quarter",
                         title="Efficiency Mapping: Volume vs. CPU",
                         labels={"Total Sales Qty": "Sales Volume", "CPU_USD": "Unit Cost (USD)"},
                         color="CPU_USD", color_continuous_scale="RdYlGn_r")
    fig_quad.add_hline(y=df_26["CPU_USD"].mean(), line_dash="dash", line_color="gray")
    fig_quad.add_vline(x=df_26["Total Sales Qty"].mean(), line_dash="dash", line_color="gray")
    fig_quad.update_traces(textposition='top center', marker=dict(size=12))
    st.plotly_chart(fig_quad, use_container_width=True)

with ev_col2:
    fig_scissors = go.Figure()
    fig_scissors.add_trace(go.Bar(x=quarters_26, y=df_26["Total Sales Qty"], name="Volume", marker_color="lightblue", opacity=0.6))
    fig_scissors.add_trace(go.Scatter(x=quarters_26, y=df_26["CPU_USD"], name="Unit Cost", yaxis="y2", line=dict(color='red', width=4)))
    fig_scissors.update_layout(title="Cost-Volume Scissors Trend (2026)",
                             yaxis=dict(title="Volume"),
                             yaxis2=dict(title="Unit Cost (USD)", overlaying='y', side='right', range=[0.3, 0.45]))
    st.plotly_chart(fig_scissors, use_container_width=True)

st.markdown("""
**Audit Observations based on above Charts:**
* **Q1'26 Benchmark:** Peak Volume successfully drives Unit Cost to minimum levels ($0.34).
* **Q2/Q3 Risk Area:** Significant volume drop forces Unit Cost to peak at $0.37+ due to fixed cost under-absorption.
""")

recommendations = [
    {"Category": "Labor Cost - Operator", "Red Flags": "High Q1'26 forecast despite new equipment; efficiency gains not visible.", "Proposed Action": "Audit OT rates and verify equipment throughput capacity vs actual output."},
    {"Category": "Total Sales Qty", "Red Flags": "Q1'26 growth deviates from historical seasonal trends.", "Proposed Action": "Confirm business drivers for Q1 peak or adjust forecast to align with historical seasonality."},
    {"Category": "Travel & Transport", "Red Flags": "Static annual values suggest budget placeholders rather than activity-based planning.", "Proposed Action": "Request itemized 2026 travel plan to justify fixed spending."},
    {"Category": "Maintenance", "Red Flags": "Static at $250 despite increasing depreciation and production volume.", "Proposed Action": "Verify risk of equipment failure due to under-budgeted maintenance amid higher usage."}
]

st.table(pd.DataFrame(recommendations))

# Export
csv = pd.DataFrame(recommendations).to_csv(index=False).encode('utf-8')
st.download_button(label="📥 Download Recommendations as CSV", data=csv, file_name='factory_audit_recommendations.csv', mime='text/csv')
