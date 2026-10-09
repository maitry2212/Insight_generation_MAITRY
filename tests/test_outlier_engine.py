import pytest
import pandas as pd
from src.outlier_engine import detect_outliers_iqr, detect_outliers_zscore, detect_outliers


def test_detect_outliers_iqr():
    # 6 districts with 1 extreme outlier (Mehsana at 42 vs ~85)
    data = {
        'month': ['2026-08'] * 6,
        'district': ['Ahmedabad', 'Surat', 'Vadodara', 'Rajkot', 'Mehsana', 'Bhavnagar'],
        'anc_coverage': [69.0, 83.0, 91.0, 80.0, 42.0, 79.0],
        'high_risk_cases': [13.0, 12.0, 6.0, 14.0, 28.0, 15.0]
    }
    df = pd.DataFrame(data)
    outliers = detect_outliers_iqr(df, iqr_multiplier=1.5, indicators=['anc_coverage', 'high_risk_cases'])
    
    anc_outliers = [o for o in outliers if o['indicator'] == 'anc_coverage']
    assert len(anc_outliers) >= 1
    assert any(o['district'] == 'Mehsana' and o['bound_breached'] == 'lower' for o in anc_outliers)

    hr_outliers = [o for o in outliers if o['indicator'] == 'high_risk_cases']
    assert any(o['district'] == 'Mehsana' and o['bound_breached'] == 'upper' for o in hr_outliers)


def test_detect_outliers_zscore_constant_column():
    data = {
        'month': ['2026-08'] * 4,
        'district': ['D1', 'D2', 'D3', 'D4'],
        'anc_coverage': [80.0, 80.0, 80.0, 80.0]
    }
    df = pd.DataFrame(data)
    outliers = detect_outliers_zscore(df, z_threshold=3.0, indicators=['anc_coverage'])
    assert len(outliers) == 0
