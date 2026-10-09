"""
Module 4: Correlation Analysis Engine
Computes Pearson correlation matrix across numerical healthcare indicators,
flags indicator pairs meeting configurable correlation thresholds (|r| >= 0.70 default),
and attaches small-sample fragility diagnostics.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from src.data_loader import NUMERIC_INDICATORS


def compute_correlation_matrix(
    df: pd.DataFrame,
    indicators: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Computes standard Pearson correlation matrix across numeric indicator columns.

    Args:
        df: Input DataFrame containing numeric indicators.
        indicators: Specific numeric indicator columns to correlate. Defaults to NUMERIC_INDICATORS.

    Returns:
        Tuple of (correlation_matrix: pd.DataFrame, metadata: Dict[str, Any]).
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(), {'sample_size': 0, 'is_small_sample': True, 'warning': 'No data available.'}

    if indicators is None:
        indicators = [c for c in NUMERIC_INDICATORS if c in df.columns]

    # Subset valid numerical columns with dropna
    valid_df = df[indicators].dropna()
    n_samples = len(valid_df)

    if n_samples < 2:
        return pd.DataFrame(index=indicators, columns=indicators), {
            'sample_size': n_samples,
            'is_small_sample': True,
            'warning': 'Insufficient data points (N < 2) to compute correlation.'
        }

    corr_matrix = valid_df.corr(method='pearson')

    is_small_sample = n_samples < 30
    warning_msg = (
        f"Small sample size warning (N = {n_samples} rows). Pearson correlation coefficients "
        "may be unstable and sensitive to individual district outliers. Interpret with caution."
        if is_small_sample else ""
    )

    metadata = {
        'sample_size': n_samples,
        'is_small_sample': is_small_sample,
        'warning': warning_msg,
        'causation_disclaimer': "Correlation indicates statistical co-movement, not a direct causal relationship."
    }

    return corr_matrix, metadata


def flag_strong_correlations(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.70,
    sample_size: int = 0
) -> List[Dict[str, Any]]:
    """
    Extracts distinct pairs of indicators whose absolute Pearson correlation |r| >= threshold.

    Args:
        corr_matrix: Symmetric correlation matrix DataFrame.
        threshold: Absolute correlation threshold (default: 0.70).
        sample_size: Number of observation rows used in the correlation.

    Returns:
        List of flagged correlation pair dictionaries.
    """
    if corr_matrix is None or corr_matrix.empty:
        return []

    flagged_pairs = []
    columns = corr_matrix.columns.tolist()

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col_x = columns[i]
            col_y = columns[j]
            r_val = corr_matrix.loc[col_x, col_y]

            if pd.isna(r_val):
                continue

            r_val = float(r_val)
            abs_r = abs(r_val)

            if abs_r >= threshold:
                direction = "positive" if r_val > 0 else "negative"
                strength = "Very Strong" if abs_r >= 0.85 else "Strong"

                flagged_pairs.append({
                    'indicator_x': col_x,
                    'indicator_y': col_y,
                    'indicator_pair': f"{col_x}:{col_y}",
                    'r_value': round(r_val, 3),
                    'abs_r_value': round(abs_r, 3),
                    'direction': direction,
                    'strength': strength,
                    'threshold_used': float(threshold),
                    'sample_size': sample_size,
                    'is_small_sample': sample_size < 30
                })

    # Sort by absolute correlation strength descending
    flagged_pairs.sort(key=lambda x: x['abs_r_value'], reverse=True)
    return flagged_pairs
