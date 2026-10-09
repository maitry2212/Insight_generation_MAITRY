import pytest
import pandas as pd
from src.data_loader import load_and_preprocess_csv
from src.insight_generator import generate_all_insights


def test_generate_all_insights_structure():
    df, _ = load_and_preprocess_csv('data/sample_healthcare_data.csv')
    insights_df = generate_all_insights(
        df,
        trend_threshold_pct=10.0,
        outlier_method="iqr",
        corr_threshold=0.70
    )
    
    # Verify non-empty
    assert len(insights_df) > 0

    # Verify mandatory 9 fields
    mandatory_fields = [
        'insight_id', 'type', 'indicator', 'entity', 'period',
        'metric', 'change', 'severity', 'explanation'
    ]
    for field in mandatory_fields:
        assert field in insights_df.columns
        assert not insights_df[field].isnull().any()

    # Verify insight_id format
    assert all(str(iid).startswith("INS-") for iid in insights_df['insight_id'])

    # Verify severity values
    allowed_severities = {'Low', 'Medium', 'High'}
    assert set(insights_df['severity'].unique()).issubset(allowed_severities)

    # Verify dynamic templating (contains valid numeric values and entity names)
    sample_expl = insights_df['explanation'].tolist()
    assert any("Ahmedabad" in exp for exp in sample_expl)
    assert any("Mehsana" in exp for exp in sample_expl)
