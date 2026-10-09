import pytest
import pandas as pd
from src.data_loader import load_and_preprocess_csv
from src.insight_generator import generate_all_insights
from src.correlation_engine import compute_correlation_matrix
from src.visualizer import (
    plot_severity_distribution,
    plot_correlation_heatmap,
    plot_district_time_series,
    plot_indicator_distribution
)


def test_visualizer_figures_render_without_errors():
    df, _ = load_and_preprocess_csv('data/sample_healthcare_data.csv')
    insights_df = generate_all_insights(df)
    corr_matrix, _ = compute_correlation_matrix(df)

    # Severity distribution
    fig_sev = plot_severity_distribution(insights_df)
    assert fig_sev is not None
    assert fig_sev.layout.paper_bgcolor == "rgba(0,0,0,0)"

    # Correlation heatmap
    fig_corr = plot_correlation_heatmap(corr_matrix)
    assert fig_corr is not None
    assert fig_corr.layout.paper_bgcolor == "rgba(0,0,0,0)"

    # District time series
    fig_line = plot_district_time_series(df, 'anc_coverage', ['Ahmedabad', 'Mehsana'])
    assert fig_line is not None
    assert fig_line.layout.paper_bgcolor == "rgba(0,0,0,0)"

    # Indicator distribution
    fig_box = plot_indicator_distribution(df, 'anc_coverage')
    assert fig_box is not None
    assert fig_box.layout.paper_bgcolor == "rgba(0,0,0,0)"
