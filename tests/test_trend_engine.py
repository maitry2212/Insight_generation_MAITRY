import pytest
import pandas as pd
from src.trend_engine import detect_trends, compute_mom_percentage_change


def test_compute_mom_percentage_change_normal():
    change, is_zero = compute_mom_percentage_change(85.0, 69.0)
    assert round(change, 2) == -18.82
    assert not is_zero


def test_compute_mom_percentage_change_zero_base():
    change, is_zero = compute_mom_percentage_change(0.0, 10.0)
    assert change == 100.0
    assert is_zero


def test_detect_trends_flagging():
    data = {
        'month': ['2026-07', '2026-08', '2026-07', '2026-08'],
        'district': ['Ahmedabad', 'Ahmedabad', 'Surat', 'Surat'],
        'anc_coverage': [85.0, 69.0, 81.0, 83.0],
        'institutional_delivery': [91.0, 90.0, 87.0, 89.0],
        'immunization': [93.0, 92.0, 91.0, 92.0],
        'high_risk_cases': [10.0, 13.0, 13.0, 12.0]
    }
    df = pd.DataFrame(data)
    trends = detect_trends(df, threshold_pct=10.0)
    
    ahmedabad_anc = [t for t in trends if t['district'] == 'Ahmedabad' and t['indicator'] == 'anc_coverage'][0]
    assert ahmedabad_anc['is_significant'] is True
    assert ahmedabad_anc['change_pct'] == -18.82
    assert ahmedabad_anc['direction'] == 'dropped'

    surat_anc = [t for t in trends if t['district'] == 'Surat' and t['indicator'] == 'anc_coverage'][0]
    assert surat_anc['is_significant'] is False
    assert surat_anc['change_pct'] == 2.47
