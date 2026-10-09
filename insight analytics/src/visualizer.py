"""
Visualization Suite
Generates interactive Plotly figures with modern styling and responsive layouts.
"""

from typing import List, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.data_loader import INDICATOR_METADATA


SEVERITY_COLORS = {
    'High': '#EF4444',     # Vibrant Crimson Red
    'Medium': '#F59E0B',   # Warm Amber
    'Low': '#3B82F6'       # Electric Blue
}


def plot_severity_distribution(insights_df: pd.DataFrame) -> go.Figure:
    """
    Creates a bar chart displaying count of insights broken down by severity level.
    """
    if insights_df is None or insights_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No insights available to plot", showarrow=False, font=dict(size=14))
        fig.update_layout(template="plotly_white", height=320)
        return fig

    # Count by severity ensuring all three categories exist
    severity_order = ['High', 'Medium', 'Low']
    counts_series = insights_df['severity'].value_counts()
    
    counts_df = pd.DataFrame({
        'Severity': severity_order,
        'Count': [counts_series.get(s, 0) for s in severity_order],
        'Color': [SEVERITY_COLORS[s] for s in severity_order]
    })

    fig = go.Figure(data=[
        go.Bar(
            x=counts_df['Severity'],
            y=counts_df['Count'],
            marker_color=counts_df['Color'],
            text=counts_df['Count'],
            textposition='auto',
            textfont=dict(size=14, color='white', family='Arial Black'),
            hoverinfo='x+y',
            width=0.45
        )
    ])

    fig.update_layout(
        title=dict(text="<b>Insights by Severity Level</b>", font=dict(size=16, color="#1E293B")),
        xaxis_title="<b>Severity Tier</b>",
        yaxis_title="<b>Count of Insights</b>",
        template="plotly_white",
        height=320,
        margin=dict(l=40, r=20, t=50, b=40),
        plot_bgcolor="rgba(248, 250, 252, 0.5)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def plot_correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """
    Generates an interactive annotated heatmap for the Pearson correlation matrix.
    """
    if corr_matrix is None or corr_matrix.empty:
        fig = go.Figure()
        fig.add_annotation(text="No correlation data available", showarrow=False, font=dict(size=14))
        fig.update_layout(template="plotly_white", height=360)
        return fig

    labels = [INDICATOR_METADATA.get(c, {}).get('label', c) for c in corr_matrix.columns]
    z_values = corr_matrix.values

    # Format text labels with 2 decimal places
    text_annotations = [[f"{val:.2f}" if not pd.isna(val) else "N/A" for val in row] for row in z_values]

    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=labels,
        y=labels,
        text=text_annotations,
        texttemplate="%{text}",
        textfont={"size": 13, "color": "white", "family": "Inter, sans-serif"},
        colorscale='RdBu_r',  # Red is positive, Blue is negative
        zmin=-1.0,
        zmax=1.0,
        colorbar=dict(
            title=dict(text="Pearson <i>r</i>", font=dict(size=12)),
            thickness=14,
            len=0.9
        ),
        hoverongaps=False
    ))

    fig.update_layout(
        title=dict(text="<b>Pearson Correlation Heatmap</b>", font=dict(size=16, color="#1E293B")),
        template="plotly_white",
        height=360,
        margin=dict(l=50, r=20, t=50, b=50),
        xaxis=dict(tickangle=-25),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def plot_district_time_series(
    df: pd.DataFrame,
    indicator: str,
    selected_districts: Optional[List[str]] = None
) -> go.Figure:
    """
    Plots multi-line time series chart comparing districts over time for a selected indicator.
    """
    if df is None or df.empty or indicator not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No time-series data available", showarrow=False, font=dict(size=14))
        fig.update_layout(template="plotly_white", height=380)
        return fig

    df_filtered = df.copy()
    if selected_districts:
        df_filtered = df_filtered[df_filtered['district'].isin(selected_districts)]

    df_filtered = df_filtered.sort_values(by=['month', 'district'])

    label = INDICATOR_METADATA.get(indicator, {}).get('label', indicator)
    unit = INDICATOR_METADATA.get(indicator, {}).get('unit', '')

    fig = px.line(
        df_filtered,
        x='month',
        y=indicator,
        color='district',
        markers=True,
        labels={'month': 'Month', indicator: f"{label} ({unit})", 'district': 'District'},
        title=f"<b>{label} Trajectory by District</b>",
        color_discrete_sequence=px.colors.qualitative.Prism
    )

    fig.update_traces(
        line=dict(width=2.5),
        marker=dict(size=8, symbol='circle'),
        hovertemplate="<b>%{data.name}</b><br>Month: %{x}<br>Value: %{y:.1f}" + unit + "<extra></extra>"
    )

    fig.update_layout(
        template="plotly_white",
        height=380,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title_text=""
        ),
        margin=dict(l=40, r=20, t=60, b=40),
        plot_bgcolor="rgba(248, 250, 252, 0.6)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def plot_indicator_distribution(df: pd.DataFrame, indicator: str) -> go.Figure:
    """
    Creates box plots showing distribution and potential outliers for an indicator across months.
    """
    if df is None or df.empty or indicator not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No distribution data", showarrow=False)
        return fig

    label = INDICATOR_METADATA.get(indicator, {}).get('label', indicator)
    unit = INDICATOR_METADATA.get(indicator, {}).get('unit', '')

    fig = px.box(
        df,
        x='month',
        y=indicator,
        points="all",
        hover_data=['district'],
        title=f"<b>Cross-Sectional Distribution: {label}</b>",
        color_discrete_sequence=['#4F46E5']
    )

    fig.update_layout(
        template="plotly_white",
        height=340,
        yaxis_title=f"{label} ({unit})",
        xaxis_title="Month",
        margin=dict(l=40, r=20, t=50, b=40),
        plot_bgcolor="rgba(248, 250, 252, 0.6)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig
