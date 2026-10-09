"""
Module 5: Automated Insight Generation Engine
Synthesizes analytical results from trends, outliers, correlations, and benchmark breaches
into standardized, structured insight objects with data-driven severity and dynamic templated copy.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
from src.data_loader import INDICATOR_METADATA, NUMERIC_INDICATORS
from src.trend_engine import detect_trends
from src.outlier_engine import detect_outliers
from src.correlation_engine import compute_correlation_matrix, flag_strong_correlations
from src.severity_classifier import (
    classify_trend_severity,
    classify_outlier_zscore_severity,
    classify_outlier_iqr_severity,
    classify_correlation_severity,
    classify_threshold_breach_severity
)


def _get_label(indicator_key: str) -> str:
    """Helper to retrieve human-readable metric name or fallback to key."""
    if indicator_key in INDICATOR_METADATA:
        return INDICATOR_METADATA[indicator_key]['label']
    return indicator_key.replace('_', ' ').title()


def _get_unit(indicator_key: str) -> str:
    """Helper to retrieve metric unit."""
    if indicator_key in INDICATOR_METADATA:
        return INDICATOR_METADATA[indicator_key]['unit']
    return ""


def generate_all_insights(
    df: pd.DataFrame,
    trend_threshold_pct: float = 10.0,
    outlier_method: str = "z_score",
    outlier_z_threshold: float = 3.0,
    outlier_iqr_multiplier: float = 1.5,
    outlier_group_by_period: bool = True,
    corr_threshold: float = 0.70,
    benchmark_thresholds: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Executes all analytical engines and produces a unified pandas DataFrame
    with mandatory structured fields and dynamic natural language explanations.

    Args:
        df: Input cleaned DataFrame.
        trend_threshold_pct: Trend detection cutoff percentage (default: 10.0).
        outlier_method: 'z_score' or 'iqr'.
        outlier_z_threshold: Parametric Z-score threshold (default: 3.0).
        outlier_iqr_multiplier: Tukey IQR multiplier (default: 1.5).
        outlier_group_by_period: If True, evaluates outliers cross-sectionally per month.
        corr_threshold: Pearson correlation cutoff (default: 0.70).
        benchmark_thresholds: Optional dict mapping indicator -> minimum acceptable benchmark value.

    Returns:
        DataFrame containing all generated structured insights.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=[
            'insight_id', 'type', 'indicator', 'entity', 'period',
            'metric', 'change', 'severity', 'explanation',
            'value', 'prev_value', 'change_pct'
        ])

    raw_insights: List[Dict[str, Any]] = []

    # ==========================================
    # 1. TREND INSIGHTS
    # ==========================================
    trends = detect_trends(df, threshold_pct=trend_threshold_pct)
    for t in trends:
        if not t['is_significant']:
            continue

        label = _get_label(t['indicator'])
        unit = _get_unit(t['indicator'])
        severity = classify_trend_severity(t['abs_change_pct'], base_threshold_pct=trend_threshold_pct)

        # Dynamic templated explanation (zero hardcoding)
        change_sign = "+" if t['change_pct'] > 0 else ""
        explanation = (
            f"{label} in {t['district']} {t['direction']} by {abs(t['change_pct']):.1f}% "
            f"(from {t['prev_value']}{unit} in {t['prev_period']} to {t['current_value']}{unit} in {t['period']}), "
            f"exceeding the {trend_threshold_pct:.1f}% significant-change threshold."
        )

        raw_insights.append({
            'type': 'trend',
            'indicator': t['indicator'],
            'entity': t['district'],
            'period': t['period'],
            'metric': t['current_value'],
            'change': f"{change_sign}{t['change_pct']:.1f}%",
            'severity': severity,
            'explanation': explanation,
            'value': t['current_value'],
            'prev_value': t['prev_value'],
            'change_pct': t['change_pct'],
            'sort_priority': 3 if severity == 'High' else (2 if severity == 'Medium' else 1)
        })

    # ==========================================
    # 2. OUTLIER INSIGHTS
    # ==========================================
    outliers = detect_outliers(
        df,
        method=outlier_method,
        z_threshold=outlier_z_threshold,
        iqr_multiplier=outlier_iqr_multiplier,
        group_by_period=outlier_group_by_period
    )

    for o in outliers:
        label = _get_label(o['indicator'])
        unit = _get_unit(o['indicator'])

        if o['method'] == 'z_score':
            severity = classify_outlier_zscore_severity(o['abs_z_score'], base_z_threshold=outlier_z_threshold)
            change_str = f"{o['z_score']:+.1f}σ"
            explanation = (
                f"{o['district']}'s {label} of {o['value']}{unit} in {o['period']} is "
                f"{o['abs_z_score']:.1f}σ {o['direction']} the group mean ({o['reference_mean']:.1f}{unit}), "
                f"flagging as an anomalous statistical outlier."
            )
            prev_val = o['reference_mean']
            change_pct = round(((o['value'] - prev_val) / prev_val) * 100.0, 2) if prev_val != 0 else 0.0
        else:
            severity = classify_outlier_iqr_severity(o['breach_ratio'])
            change_str = f"{o['bound_breached']} bound breach"
            explanation = (
                f"{o['district']}'s {label} of {o['value']}{unit} in {o['period']} breached the normal "
                f"IQR boundary [{o['lower_bound']:.1f}, {o['upper_bound']:.1f}] with an observed deviation of "
                f"{o['breach_ratio']:.1f}x IQR."
            )
            prev_val = o['reference_mean']
            change_pct = round(((o['value'] - prev_val) / prev_val) * 100.0, 2) if prev_val != 0 else 0.0

        raw_insights.append({
            'type': 'outlier',
            'indicator': o['indicator'],
            'entity': o['district'],
            'period': o['period'],
            'metric': o['value'],
            'change': change_str,
            'severity': severity,
            'explanation': explanation,
            'value': o['value'],
            'prev_value': prev_val,
            'change_pct': change_pct,
            'sort_priority': 3 if severity == 'High' else (2 if severity == 'Medium' else 1)
        })

    # ==========================================
    # 3. CORRELATION INSIGHTS
    # ==========================================
    corr_matrix, meta = compute_correlation_matrix(df)
    flagged_corr = flag_strong_correlations(corr_matrix, threshold=corr_threshold, sample_size=meta['sample_size'])

    # Determine period span for correlation
    months = sorted(df['month'].dropna().unique().tolist())
    period_str = f"{months[0]}..{months[-1]}" if len(months) > 1 else (months[0] if months else "All Periods")

    for c in flagged_corr:
        label_x = _get_label(c['indicator_x'])
        label_y = _get_label(c['indicator_y'])
        severity = classify_correlation_severity(c['abs_r_value'], base_threshold=corr_threshold)

        explanation = (
            f"Strong {c['direction']} Pearson correlation (r = {c['r_value']:+.2f}) detected between "
            f"{label_x} and {label_y} across {c['sample_size']} observations ({period_str})."
        )
        if c['is_small_sample']:
            explanation += " [Caution: Small sample size - correlation may be fragile]."

        raw_insights.append({
            'type': 'correlation',
            'indicator': c['indicator_pair'],
            'entity': 'State-wide',
            'period': period_str,
            'metric': f"r = {c['r_value']:+.2f}",
            'change': f"{c['strength']} ({c['direction']})",
            'severity': severity,
            'explanation': explanation,
            'value': c['r_value'],
            'prev_value': None,
            'change_pct': None,
            'sort_priority': 3 if severity == 'High' else (2 if severity == 'Medium' else 1)
        })

    # ==========================================
    # 4. THRESHOLD BREACH INSIGHTS (Optional Benchmarks)
    # ==========================================
    if benchmark_thresholds:
        for ind, benchmark in benchmark_thresholds.items():
            if ind not in df.columns:
                continue
            label = _get_label(ind)
            unit = _get_unit(ind)

            for _, row in df.iterrows():
                val = row[ind]
                if pd.isna(val):
                    continue
                val = float(val)

                if val < benchmark:
                    severity = classify_threshold_breach_severity(val, benchmark, is_lower_breach=True)
                    deficit = benchmark - val
                    explanation = (
                        f"{row['district']}'s {label} ({val:.1f}{unit}) in {row['month']} failed to meet the "
                        f"target benchmark of {benchmark:.1f}{unit} (Deficit: {deficit:.1f}{unit})."
                    )

                    raw_insights.append({
                        'type': 'threshold_breach',
                        'indicator': ind,
                        'entity': str(row['district']),
                        'period': str(row['month']),
                        'metric': val,
                        'change': f"-{deficit:.1f}{unit}",
                        'severity': severity,
                        'explanation': explanation,
                        'value': val,
                        'prev_value': benchmark,
                        'change_pct': round((-deficit / benchmark) * 100.0, 2) if benchmark != 0 else 0.0,
                        'sort_priority': 3 if severity == 'High' else (2 if severity == 'Medium' else 1)
                    })

    # Sort insights by severity priority descending (High -> Medium -> Low), then period descending
    raw_insights.sort(key=lambda x: (x['sort_priority'], str(x['period'])), reverse=True)

    # Assign sequential Insight IDs (INS-0001, INS-0002, ...)
    for idx, item in enumerate(raw_insights, start=1):
        item['insight_id'] = f"INS-{idx:04d}"
        if 'sort_priority' in item:
            del item['sort_priority']

    insights_df = pd.DataFrame(raw_insights)
    
    col_order = [
        'insight_id', 'type', 'indicator', 'entity', 'period',
        'metric', 'change', 'severity', 'explanation',
        'value', 'prev_value', 'change_pct'
    ]
    # Keep any extra columns at the end if present
    existing_cols = [c for c in col_order if c in insights_df.columns]
    other_cols = [c for c in insights_df.columns if c not in existing_cols]
    insights_df = insights_df[existing_cols + other_cols]

    return insights_df
