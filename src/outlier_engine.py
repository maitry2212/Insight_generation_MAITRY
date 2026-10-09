"""
Module 3: Outlier Detection Engine
Supports IQR (Interquartile Range) and Z-Score outlier detection across districts
for healthcare indicators, handling constant distributions, zero standard deviations,
and small sample sizes safely.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from src.data_loader import NUMERIC_INDICATORS


def detect_outliers_iqr(
    df: pd.DataFrame,
    iqr_multiplier: float = 1.5,
    group_by_period: bool = True,
    indicators: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Detects statistical outliers using Tukey's Interquartile Range (IQR) rule:
    Lower Bound = Q1 - (k * IQR)
    Upper Bound = Q3 + (k * IQR)

    Args:
        df: Input DataFrame containing 'month', 'district' and numeric indicators.
        iqr_multiplier: Tukey's multiplier k (default: 1.5).
        group_by_period: If True, evaluates outliers per month; otherwise across the entire dataset.
        indicators: Numeric indicators to check. Defaults to NUMERIC_INDICATORS.

    Returns:
        List of outlier dictionaries with complete statistical parameters.
    """
    if df is None or len(df) == 0:
        return []

    if indicators is None:
        indicators = [c for c in NUMERIC_INDICATORS if c in df.columns]

    outliers: List[Dict[str, Any]] = []

    if group_by_period and 'month' in df.columns:
        grouped = list(df.groupby('month'))
    else:
        grouped = [(None, df)]

    for period, group in grouped:
        for ind in indicators:
            if ind not in group.columns:
                continue

            series = pd.to_numeric(group[ind], errors='coerce').dropna()
            if len(series) < 3:
                continue

            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1

            if iqr == 0:
                # Distribution is constant or has low variance
                continue

            lower_bound = q1 - (iqr_multiplier * iqr)
            upper_bound = q3 + (iqr_multiplier * iqr)

            for _, row in group.iterrows():
                val = row[ind]
                if pd.isna(val):
                    continue
                val = float(val)

                is_outlier = (val < lower_bound) or (val > upper_bound)
                if is_outlier:
                    bound_breached = "lower" if val < lower_bound else "upper"
                    dist_from_bound = (lower_bound - val) if val < lower_bound else (val - upper_bound)
                    breach_ratio = dist_from_bound / iqr if iqr > 0 else 0.0

                    outliers.append({
                        'district': str(row['district']),
                        'indicator': str(ind),
                        'period': str(row['month']) if 'month' in row else "all_time",
                        'value': round(val, 2),
                        'method': 'iqr',
                        'q1': round(q1, 2),
                        'q3': round(q3, 2),
                        'iqr': round(iqr, 2),
                        'lower_bound': round(lower_bound, 2),
                        'upper_bound': round(upper_bound, 2),
                        'bound_breached': bound_breached,
                        'breach_ratio': round(float(breach_ratio), 2),
                        'reference_mean': round(float(series.mean()), 2),
                        'reference_std': round(float(series.std(ddof=1)), 2) if len(series) > 1 else 0.0,
                        'is_outlier': True
                    })

    return outliers


def detect_outliers_zscore(
    df: pd.DataFrame,
    z_threshold: float = 3.0,
    group_by_period: bool = True,
    indicators: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Detects statistical outliers using Standardized Z-scores:
    Z = (x - mean) / std
    Flagged if |Z| >= z_threshold (default: 3.0).

    Args:
        df: Input DataFrame.
        z_threshold: Standard deviation threshold cutoff.
        group_by_period: If True, computes per-month Z-scores.
        indicators: List of numeric columns.

    Returns:
        List of outlier dictionaries with Z-score statistics.
    """
    if df is None or len(df) == 0:
        return []

    if indicators is None:
        indicators = [c for c in NUMERIC_INDICATORS if c in df.columns]

    outliers: List[Dict[str, Any]] = []

    if group_by_period and 'month' in df.columns:
        grouped = list(df.groupby('month'))
    else:
        grouped = [(None, df)]

    for period, group in grouped:
        for ind in indicators:
            if ind not in group.columns:
                continue

            series = pd.to_numeric(group[ind], errors='coerce').dropna()
            n = len(series)
            if n < 3:
                continue

            mean_val = float(series.mean())
            std_val = float(series.std(ddof=1)) if n > 1 else 0.0

            if std_val == 0.0:
                continue

            for _, row in group.iterrows():
                val = row[ind]
                if pd.isna(val):
                    continue
                val = float(val)

                z_score = (val - mean_val) / std_val
                abs_z = abs(z_score)

                if abs_z >= z_threshold:
                    direction = "below" if z_score < 0 else "above"
                    outliers.append({
                        'district': str(row['district']),
                        'indicator': str(ind),
                        'period': str(row['month']) if 'month' in row else "all_time",
                        'value': round(val, 2),
                        'method': 'z_score',
                        'z_score': round(float(z_score), 2),
                        'abs_z_score': round(float(abs_z), 2),
                        'reference_mean': round(mean_val, 2),
                        'reference_std': round(std_val, 2),
                        'direction': direction,
                        'threshold_used': float(z_threshold),
                        'is_outlier': True
                    })

    return outliers


def detect_outliers(
    df: pd.DataFrame,
    method: str = "z_score",
    z_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
    group_by_period: bool = True,
    indicators: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Unified entry point for outlier detection supporting both 'z_score' and 'iqr' methods.
    """
    if method.lower() == "iqr":
        return detect_outliers_iqr(
            df=df,
            iqr_multiplier=iqr_multiplier,
            group_by_period=group_by_period,
            indicators=indicators
        )
    else:
        return detect_outliers_zscore(
            df=df,
            z_threshold=z_threshold,
            group_by_period=group_by_period,
            indicators=indicators
        )
