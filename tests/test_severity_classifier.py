import pytest
from src.severity_classifier import (
    classify_trend_severity,
    classify_outlier_zscore_severity,
    classify_outlier_iqr_severity,
    classify_correlation_severity
)


def test_classify_trend_severity():
    base_thresh = 10.0
    assert classify_trend_severity(8.0, base_thresh) == "Low"
    assert classify_trend_severity(12.0, base_thresh) == "Low"
    assert classify_trend_severity(18.0, base_thresh) == "Medium"
    assert classify_trend_severity(30.0, base_thresh) == "High"


def test_classify_outlier_zscore_severity():
    base_z = 3.0
    assert classify_outlier_zscore_severity(3.2, base_z) == "Low"
    assert classify_outlier_zscore_severity(3.7, base_z) == "Medium"
    assert classify_outlier_zscore_severity(4.5, base_z) == "High"


def test_classify_correlation_severity():
    base_r = 0.70
    assert classify_correlation_severity(0.75, base_r) == "Low"
    assert classify_correlation_severity(0.85, base_r) == "Medium"
    assert classify_correlation_severity(0.95, base_r) == "High"
