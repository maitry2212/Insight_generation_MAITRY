"""
Module 1: Data Loading, Validation and Profiling
Provides robust CSV ingestion, schema validation, type casting,
missing value diagnostics, and data profiling summaries.
"""

from typing import Tuple, Dict, Any, List, Union
import io
import pandas as pd
import numpy as np

REQUIRED_COLUMNS = [
    'month',
    'district',
    'anc_coverage',
    'institutional_delivery',
    'immunization',
    'high_risk_cases'
]

NUMERIC_INDICATORS = [
    'anc_coverage',
    'institutional_delivery',
    'immunization',
    'high_risk_cases'
]

INDICATOR_METADATA = {
    'anc_coverage': {
        'label': 'ANC Coverage',
        'unit': '%',
        'is_rate': True,
        'description': 'Antenatal care registration coverage percentage'
    },
    'institutional_delivery': {
        'label': 'Institutional Delivery',
        'unit': '%',
        'is_rate': True,
        'description': 'Percentage of deliveries in health institutions'
    },
    'immunization': {
        'label': 'Immunization Rate',
        'unit': '%',
        'is_rate': True,
        'description': 'Full immunization coverage percentage'
    },
    'high_risk_cases': {
        'label': 'High Risk Cases',
        'unit': 'cases',
        'is_rate': False,
        'description': 'Count of identified high risk pregnancy cases'
    }
}


def validate_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validates if the DataFrame contains all mandatory columns and valid data structure.

    Args:
        df: Input pandas DataFrame.

    Returns:
        Tuple of (is_valid: bool, error_messages: List[str]).
    """
    errors = []
    if df is None or not isinstance(df, pd.DataFrame):
        return False, ["Input is not a valid pandas DataFrame."]

    if len(df) == 0:
        return False, ["The uploaded CSV file is empty (0 rows)."]

    # Normalize column names (strip whitespace and lower-case)
    df.columns = [str(col).strip().lower() for col in df.columns]

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing mandatory columns: {', '.join(missing_cols)}")

    return len(errors) == 0, errors


def get_dataframe_info_str(df: pd.DataFrame) -> str:
    """
    Returns a formatted string summary equivalent to df.info().
    """
    buf = io.StringIO()
    df.info(buf=buf)
    return buf.getvalue()


def load_and_preprocess_csv(
    file_or_path: Union[str, io.BytesIO, io.StringIO]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads CSV, validates schema, converts data types, detects duplicates,
    and returns cleaned DataFrame along with comprehensive diagnostic metadata.

    Args:
        file_or_path: File path or uploaded file buffer.

    Returns:
        Tuple of (cleaned_df: pd.DataFrame, diagnostics: Dict[str, Any]).

    Raises:
        ValueError: If file cannot be parsed or schema validation fails.
    """
    try:
        df = pd.read_csv(file_or_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {str(e)}")

    is_valid, errors = validate_schema(df)
    if not is_valid:
        raise ValueError("; ".join(errors))

    # Standardize column headers
    df.columns = [str(col).strip().lower() for col in df.columns]

    # Pre-cleaning diagnostics
    raw_row_count = len(df)
    raw_missing_counts = {col: int(df[col].isna().sum()) for col in REQUIRED_COLUMNS if col in df.columns}

    # Clean district and month strings
    df['district'] = df['district'].astype(str).str.strip()
    df['month'] = df['month'].astype(str).str.strip()

    # Numeric conversion with coercion tracking
    invalid_numeric_counts = {}
    for col in NUMERIC_INDICATORS:
        original_non_nulls = df[col].notna().sum()
        df[col] = pd.to_numeric(df[col], errors='coerce')
        coerced_nan = int(original_non_nulls - df[col].notna().sum())
        invalid_numeric_counts[col] = coerced_nan

    # Check duplicates on (district, month)
    duplicate_mask = df.duplicated(subset=['district', 'month'], keep=False)
    duplicate_count = int(duplicate_mask.sum())
    duplicate_rows = df[duplicate_mask].to_dict(orient='records') if duplicate_count > 0 else []

    # Sort deterministically by month then district
    df = df.sort_values(by=['month', 'district']).reset_index(drop=True)

    # Compile diagnostic report
    diagnostics = {
        'total_rows': len(df),
        'raw_row_count': raw_row_count,
        'unique_districts': int(df['district'].nunique()),
        'districts_list': sorted(df['district'].unique().tolist()),
        'unique_months': int(df['month'].nunique()),
        'months_list': sorted(df['month'].unique().tolist()),
        'missing_counts': raw_missing_counts,
        'has_missing_values': any(v > 0 for v in raw_missing_counts.values()),
        'invalid_numeric_counts': invalid_numeric_counts,
        'has_invalid_numerics': any(v > 0 for v in invalid_numeric_counts.values()),
        'duplicate_records_count': duplicate_count,
        'duplicate_records': duplicate_rows,
        'dataframe_info': get_dataframe_info_str(df)
    }

    return df, diagnostics
