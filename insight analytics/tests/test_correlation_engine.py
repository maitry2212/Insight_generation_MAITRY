import pytest
import pandas as pd
from src.correlation_engine import compute_correlation_matrix, flag_strong_correlations


def test_correlation_computation_and_flagging():
    # Perfectly correlated x and y
    data = {
        'anc_coverage': [10.0, 20.0, 30.0, 40.0],
        'institutional_delivery': [20.0, 40.0, 60.0, 80.0],
        'immunization': [100.0, 90.0, 80.0, 70.0],
        'high_risk_cases': [5.0, 5.0, 5.0, 5.0]  # Zero variance
    }
    df = pd.DataFrame(data)
    matrix, meta = compute_correlation_matrix(df)
    
    assert round(matrix.loc['anc_coverage', 'institutional_delivery'], 2) == 1.00
    assert round(matrix.loc['anc_coverage', 'immunization'], 2) == -1.00
    assert meta['sample_size'] == 4
    assert meta['is_small_sample'] is True

    flagged = flag_strong_correlations(matrix, threshold=0.70, sample_size=4)
    pairs = [f['indicator_pair'] for f in flagged]
    assert 'anc_coverage:institutional_delivery' in pairs
    assert 'anc_coverage:immunization' in pairs
