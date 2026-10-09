import pytest
import io
import pandas as pd
from src.data_loader import load_and_preprocess_csv, validate_schema, REQUIRED_COLUMNS


def test_valid_csv_loading():
    valid_csv = """month,district,anc_coverage,institutional_delivery,immunization,high_risk_cases
2026-07,Ahmedabad,85,91,93,10
2026-08,Ahmedabad,69,90,92,13
"""
    df, diag = load_and_preprocess_csv(io.StringIO(valid_csv))
    assert len(df) == 2
    assert diag['total_rows'] == 2
    assert diag['unique_districts'] == 1
    assert diag['unique_months'] == 2
    assert diag['duplicate_records_count'] == 0


def test_missing_mandatory_columns():
    invalid_csv = """month,district,anc_coverage
2026-07,Ahmedabad,85
"""
    with pytest.raises(ValueError) as excinfo:
        load_and_preprocess_csv(io.StringIO(invalid_csv))
    assert "Missing mandatory columns" in str(excinfo.value)


def test_duplicate_district_month():
    dup_csv = """month,district,anc_coverage,institutional_delivery,immunization,high_risk_cases
2026-07,Ahmedabad,85,91,93,10
2026-07,Ahmedabad,86,92,94,11
"""
    df, diag = load_and_preprocess_csv(io.StringIO(dup_csv))
    assert diag['duplicate_records_count'] == 2
