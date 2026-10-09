"""
Automated Insight Generation — AI/ML
Interactive Streamlit Auto-Analytics Dashboard
"""

import os
import io
import streamlit as st
import pandas as pd
import numpy as np

from src.data_loader import (
    load_and_preprocess_csv,
    REQUIRED_COLUMNS,
    NUMERIC_INDICATORS,
    INDICATOR_METADATA
)
from src.trend_engine import detect_trends
from src.outlier_engine import detect_outliers
from src.correlation_engine import compute_correlation_matrix, flag_strong_correlations
from src.insight_generator import generate_all_insights
from src.visualizer import (
    plot_severity_distribution,
    plot_correlation_heatmap,
    plot_district_time_series,
    plot_indicator_distribution,
    SEVERITY_COLORS
)

# Set page configuration
st.set_page_config(
    page_title="Automated Insight Generation — AI/ML",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern, Polished Aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
        padding: 24px 32px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(30, 27, 75, 0.2);
    }
    
    .main-header h1 {
        color: #FFFFFF !important;
        font-size: 28px;
        font-weight: 700;
        margin: 0;
        padding-bottom: 6px;
    }
    
    .main-header p {
        color: #C7D2FE;
        font-size: 14px;
        margin: 0;
    }

    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        transition: transform 0.15s ease-in-out;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.08);
    }
    .kpi-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        font-weight: 600;
    }
    .kpi-value {
        font-size: 26px;
        font-weight: 700;
        color: #0F172A;
        margin-top: 4px;
    }
    
    .insight-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #CBD5E1;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .insight-card.high {
        border-left-color: #EF4444;
        background: #FEF2F2;
    }
    .insight-card.medium {
        border-left-color: #F59E0B;
        background: #FFFBEB;
    }
    .insight-card.low {
        border-left-color: #3B82F6;
        background: #EFF6FF;
    }
    
    .badge {
        display: inline-block;
        padding: 2px 8px;
        font-size: 11px;
        font-weight: 700;
        border-radius: 9999px;
        text-transform: uppercase;
    }
    .badge-high { background-color: #FEE2E2; color: #991B1B; }
    .badge-medium { background-color: #FEF3C7; color: #92400E; }
    .badge-low { background-color: #DBEAFE; color: #1E40AF; }
    .badge-type { background-color: #F1F5F9; color: #334155; }
</style>
""", unsafe_allow_html=True)


def main():
    # -------------------------------------------------------------
    # App Header
    # -------------------------------------------------------------
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Automated Insight Generation — AI/ML</h1>
        <p>Intelligent Auto-Analytics Engine for Healthcare Performance Diagnostics: Trends, Outliers, and Correlations</p>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # Sidebar Controls & File Upload
    # -------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Data & Controls")
        
        # Data Source Selection
        data_source = st.radio(
            "Select Data Source:",
            ["Default Sample Dataset", "Upload Custom CSV"],
            index=0
        )
        
        uploaded_file = None
        if data_source == "Upload Custom CSV":
            uploaded_file = st.file_uploader("Upload District Healthcare CSV", type=["csv"])
        
        st.markdown("---")
        st.subheader("🎯 Detection Thresholds")
        
        trend_thresh = st.slider(
            "Trend % Change Threshold (θ_trend):",
            min_value=1.0,
            max_value=50.0,
            value=10.0,
            step=1.0,
            help="Minimum month-over-month absolute percentage change to flag as a significant trend."
        )
        
        outlier_method = st.radio(
            "Outlier Detection Algorithm:",
            ["Z-Score Method", "IQR Method"],
            index=1,
            help="Choose between Tukey's Interquartile Range (IQR) or Standardized Gaussian Z-Score."
        )
        
        if outlier_method == "Z-Score Method":
            z_thresh = st.slider(
                "Z-Score Cutoff (θ_Z):",
                min_value=1.5,
                max_value=4.5,
                value=3.0,
                step=0.1,
                help="Standard deviations from the mean to qualify as an outlier."
            )
            iqr_mult = 1.5
            selected_method_key = "z_score"
        else:
            iqr_mult = st.slider(
                "IQR Multiplier (k):",
                min_value=1.0,
                max_value=3.0,
                value=1.5,
                step=0.1,
                help="Tukey's IQR fence multiplier: Q1 - k*IQR or Q3 + k*IQR."
            )
            z_thresh = 3.0
            selected_method_key = "iqr"

        corr_thresh = st.slider(
            "Pearson Correlation Cutoff (|r|):",
            min_value=0.30,
            max_value=0.99,
            value=0.70,
            step=0.05,
            help="Minimum absolute correlation coefficient to flag an indicator pair."
        )

    # -------------------------------------------------------------
    # Load & Validate Data
    # -------------------------------------------------------------
    df = None
    diagnostics = {}
    
    try:
        if data_source == "Upload Custom CSV" and uploaded_file is not None:
            df, diagnostics = load_and_preprocess_csv(uploaded_file)
        else:
            sample_path = "data/sample_healthcare_data.csv"
            if not os.path.exists(sample_path):
                sample_path = "sample_healthcare_data.csv"
            df, diagnostics = load_and_preprocess_csv(sample_path)
    except Exception as e:
        st.error(f"❌ **Data Ingestion Error:** {str(e)}")
        st.info("Please ensure the CSV contains: `month, district, anc_coverage, institutional_delivery, immunization, high_risk_cases`.")
        return

    # -------------------------------------------------------------
    # Reactive Dashboard Filters (Sidebar)
    # -------------------------------------------------------------
    with st.sidebar:
        st.markdown("---")
        st.subheader("🔍 Live Filters")
        
        all_districts = diagnostics.get('districts_list', [])
        selected_districts = st.multiselect(
            "Filter by District(s):",
            options=all_districts,
            default=all_districts
        )
        
        all_months = diagnostics.get('months_list', [])
        selected_months = st.multiselect(
            "Filter by Month(s):",
            options=all_months,
            default=all_months
        )
        
        indicator_options = list(INDICATOR_METADATA.keys())
        selected_indicators = st.multiselect(
            "Filter by Indicator(s):",
            options=indicator_options,
            format_func=lambda x: INDICATOR_METADATA[x]['label'],
            default=indicator_options
        )
        
        severity_filter = st.selectbox(
            "Filter by Severity:",
            options=["All Severities", "High", "Medium", "Low"],
            index=0
        )

        type_filter = st.selectbox(
            "Filter by Insight Type:",
            options=["All Types", "trend", "outlier", "correlation", "threshold_breach"],
            index=0
        )

    # -------------------------------------------------------------
    # Generate Insights & Compute Matrices
    # -------------------------------------------------------------
    insights_df = generate_all_insights(
        df=df,
        trend_threshold_pct=trend_thresh,
        outlier_method=selected_method_key,
        outlier_z_threshold=z_thresh,
        outlier_iqr_multiplier=iqr_mult,
        corr_threshold=corr_thresh
    )

    corr_matrix, corr_meta = compute_correlation_matrix(df)

    # -------------------------------------------------------------
    # Top KPI Metrics Strip
    # -------------------------------------------------------------
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
    
    with kpi_col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Records</div>
            <div class="kpi-value">{diagnostics.get('total_rows', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Districts Tracked</div>
            <div class="kpi-value">{diagnostics.get('unique_districts', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Months Analyzed</div>
            <div class="kpi-value">{diagnostics.get('unique_months', 0)}</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col4:
        missing_total = sum(diagnostics.get('missing_counts', {}).values())
        missing_color = "#EF4444" if missing_total > 0 else "#10B981"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Missing Values</div>
            <div class="kpi-value" style="color: {missing_color};">{missing_total}</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col5:
        high_sev_count = (insights_df['severity'] == 'High').sum() if not insights_df.empty else 0
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">High Severity Insights</div>
            <div class="kpi-value" style="color: #EF4444;">{high_sev_count}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # -------------------------------------------------------------
    # Data Quality & Validation Summary Expander
    # -------------------------------------------------------------
    with st.expander("📋 Data Health, Schema Validation & Missing Values Diagnostics", expanded=False):
        d_col1, d_col2 = st.columns([1, 1])
        with d_col1:
            st.markdown("#### Dataset Preview (`df.head()`)")
            st.dataframe(df.head(10), use_container_width=True)
            
        with d_col2:
            st.markdown("#### Missing-Value Count per Column")
            missing_df = pd.DataFrame(
                list(diagnostics.get('missing_counts', {}).items()),
                columns=['Column Name', 'Missing Count']
            )
            st.dataframe(missing_df, use_container_width=True)

        if diagnostics.get('duplicate_records_count', 0) > 0:
            st.warning(f"⚠️ Detected {diagnostics['duplicate_records_count']} duplicate records for (district, month).")

    # -------------------------------------------------------------
    # Apply UI Filters to Insights Table
    # -------------------------------------------------------------
    filtered_insights = insights_df.copy()
    if not filtered_insights.empty:
        # District filter
        if selected_districts:
            filtered_insights = filtered_insights[
                (filtered_insights['entity'].isin(selected_districts)) | 
                (filtered_insights['entity'] == 'State-wide')
            ]
        # Month/Period filter
        if selected_months:
            filtered_insights = filtered_insights[
                (filtered_insights['period'].isin(selected_months)) | 
                (filtered_insights['period'].str.contains(r'\.\.'))
            ]
        # Indicator filter
        if selected_indicators:
            pattern = '|'.join(selected_indicators)
            filtered_insights = filtered_insights[filtered_insights['indicator'].str.contains(pattern)]
        # Severity filter
        if severity_filter != "All Severities":
            filtered_insights = filtered_insights[filtered_insights['severity'] == severity_filter]
        # Type filter
        if type_filter != "All Types":
            filtered_insights = filtered_insights[filtered_insights['type'] == type_filter]

    # -------------------------------------------------------------
    # Section 1: Automated Insights Feed
    # -------------------------------------------------------------
    st.markdown("### 🔔 Automated Strategic Insights Feed")
    
    if filtered_insights.empty:
        st.info("No insights match the current filter selection or threshold criteria.")
    else:
        # Display top 5 key highlight cards
        top_cards = filtered_insights.head(5)
        for _, ins in top_cards.iterrows():
            sev_lower = ins['severity'].lower()
            badge_class = f"badge-{sev_lower}"
            st.markdown(f"""
            <div class="insight-card {sev_lower}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <div>
                        <span class="badge badge-type">{ins['insight_id']}</span>
                        <span class="badge badge-type">{ins['type'].upper()}</span>
                        <span class="badge {badge_class}">{ins['severity']}</span>
                        <strong style="margin-left: 8px; color: #1E293B;">{ins['entity']} — {ins['indicator']}</strong>
                    </div>
                    <span style="font-size: 12px; color: #64748B; font-weight: 500;">Period: {ins['period']}</span>
                </div>
                <div style="color: #334155; font-size: 14.5px; line-height: 1.5;">
                    {ins['explanation']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # Section 2: Visualizations Suite
    # -------------------------------------------------------------
    st.markdown("### 📊 Interactive Visual Analytics")
    
    v_col1, v_col2 = st.columns([1, 1])
    
    with v_col1:
        st.plotly_chart(plot_severity_distribution(filtered_insights), use_container_width=True)

    with v_col2:
        st.plotly_chart(plot_correlation_heatmap(corr_matrix), use_container_width=True)
        if corr_meta.get('is_small_sample', False):
            st.caption(f"⚠️ **Note:** {corr_meta.get('warning')}")

    v_col3, v_col4 = st.columns([1.3, 1])
    
    with v_col3:
        chart_ind = st.selectbox(
            "Select Indicator for District Trajectory Comparison:",
            options=NUMERIC_INDICATORS,
            format_func=lambda x: INDICATOR_METADATA[x]['label'],
            index=0
        )
        st.plotly_chart(plot_district_time_series(df, chart_ind, selected_districts), use_container_width=True)
        
    with v_col4:
        st.plotly_chart(plot_indicator_distribution(df, chart_ind), use_container_width=True)

    # -------------------------------------------------------------
    # Section 3: Structured Insights Table & Export
    # -------------------------------------------------------------
    st.markdown("### 📑 Structured Insights Table")
    
    # Render interactive DataFrame
    display_cols = ['insight_id', 'type', 'indicator', 'entity', 'period', 'metric', 'change', 'severity', 'explanation']
    available_display = [c for c in display_cols if c in filtered_insights.columns]
    
    st.dataframe(
        filtered_insights[available_display] if not filtered_insights.empty else pd.DataFrame(columns=display_cols),
        use_container_width=True,
        hide_index=True
    )

    # Download Buttons
    exp_col1, exp_col2, _ = st.columns([1, 1, 2])
    
    with exp_col1:
        csv_insights = filtered_insights.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Insights CSV",
            data=csv_insights,
            file_name="automated_healthcare_insights.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    with exp_col2:
        csv_corr = corr_matrix.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Download Correlation Matrix CSV",
            data=csv_corr,
            file_name="pearson_correlation_matrix.csv",
            mime="text/csv",
            use_container_width=True
        )


if __name__ == "__main__":
    main()
