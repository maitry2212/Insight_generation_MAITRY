"""
Module 2: Trend Detection Engine
Computes Month-over-Month (MoM) percentage change for each (district, indicator) pair,
supports configurable thresholding (default 10%), and handles edge cases such as
zero denominators, missing data points, and insufficient observation counts.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from src.data_loader import NUMERIC_INDICATORS


def compute_mom_percentage_change(prev_val: float, curr_val: float) -> Tuple[Optional[float], bool]:
    """
    Helper function to safely compute percentage change between two consecutive values.
    Returns (change_pct, is_zero_base).
    """
    if pd.isna(prev_val) or pd.isna(curr_val):
        return None, False

    prev_val = float(prev_val)
    curr_val = float(curr_val)

    if prev_val == 0.0:
        if curr_val == 0.0:
            return 0.0, True
        else:
            # Undefined division by zero: represent as 100% surge with zero-base flag
            return 100.0 if curr_val > 0 else -100.0, True

    change_pct = ((curr_val - prev_val) / prev_val) * 100.0
    return change_pct, False


def detect_trends(
    df: pd.DataFrame,
    threshold_pct: float = 10.0,
    indicators: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Detects significant month-over-month trends for each district and indicator.

    Args:
        df: Input DataFrame containing 'month', 'district', and indicator columns.
        threshold_pct: Absolute percentage threshold to flag significant shift (default: 10.0).
        indicators: Specific numeric indicator columns to analyze. Defaults to all 4 standard metrics.

    Returns:
        List of dictionaries containing detailed trend calculation results.
    """
    if df is None or len(df) == 0:
        return []

    if indicators is None:
        indicators = [col for col in NUMERIC_INDICATORS if col in df.columns]

    # Ensure required columns exist
    if 'district' not in df.columns or 'month' not in df.columns:
        raise ValueError("DataFrame must contain 'district' and 'month' columns.")

    # Clean and sort chronologically per district
    df_sorted = df.copy()
    df_sorted['month'] = df_sorted['month'].astype(str)
    df_sorted = df_sorted.sort_values(by=['district', 'month']).reset_index(drop=True)

    trends: List[Dict[str, Any]] = []

    for district, group in df_sorted.groupby('district'):
        group_sorted = group.sort_values(by='month')
        if len(group_sorted) < 2:
            # Insufficient observations for trend calculation
            continue

        months = group_sorted['month'].tolist()

        for ind in indicators:
            if ind not in group_sorted.columns:
                continue

            values = group_sorted[ind].tolist()

            for i in range(1, len(values)):
                prev_val = values[i - 1]
                curr_val = values[i]
                period = str(months[i])
                prev_period = str(months[i - 1])

                change_pct, is_zero_base = compute_mom_percentage_change(prev_val, curr_val)
                if change_pct is None:
                    continue

                abs_change = abs(change_pct)
                is_significant = abs_change >= threshold_pct

                if change_pct > 0:
                    direction = "increased"
                elif change_pct < 0:
                    direction = "dropped"
                else:
                    direction = "remained unchanged"

                raw_delta = float(curr_val) - float(prev_val)

                trends.append({
                    'district': str(district),
                    'indicator': str(ind),
                    'period': period,
                    'prev_period': prev_period,
                    'current_value': round(float(curr_val), 2),
                    'prev_value': round(float(prev_val), 2),
                    'raw_delta': round(raw_delta, 2),
                    'change_pct': round(float(change_pct), 2),
                    'abs_change_pct': round(float(abs_change), 2),
                    'direction': direction,
                    'is_significant': bool(is_significant),
                    'is_zero_base': is_zero_base,
                    'threshold_used': float(threshold_pct)
                })

    return trends
