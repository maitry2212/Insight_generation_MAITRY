"""
Severity Classification Module
Provides purely data-driven severity mapping (Low, Medium, High)
based on proportional thresholds and statistical distances.
"""


def classify_trend_severity(abs_change_pct: float, base_threshold_pct: float = 10.0) -> str:
    """
    Classifies trend severity based on how much the percentage change exceeds the base threshold.
    - Low: base_threshold <= abs_change < 1.5 * base_threshold (e.g. 10% - 14.9%)
    - Medium: 1.5 * base_threshold <= abs_change < 2.5 * base_threshold (e.g. 15% - 24.9%)
    - High: abs_change >= 2.5 * base_threshold (e.g. >= 25%)
    """
    if abs_change_pct < base_threshold_pct:
        return "Low"

    ratio = abs_change_pct / base_threshold_pct if base_threshold_pct > 0 else 1.0

    if ratio < 1.5:
        return "Low"
    elif ratio < 2.5:
        return "Medium"
    else:
        return "High"


def classify_outlier_zscore_severity(abs_z_score: float, base_z_threshold: float = 3.0) -> str:
    """
    Classifies Z-score outlier severity:
    - Low: base_z <= |Z| < base_z + 0.5
    - Medium: base_z + 0.5 <= |Z| < base_z + 1.2
    - High: |Z| >= base_z + 1.2 (or |Z| >= 3.5)
    """
    if abs_z_score < base_z_threshold:
        return "Low"

    diff = abs_z_score - base_z_threshold
    if diff < 0.5:
        return "Low"
    elif diff < 1.2:
        return "Medium"
    else:
        return "High"


def classify_outlier_iqr_severity(breach_ratio: float) -> str:
    """
    Classifies IQR outlier severity based on distance outside Tukey's fences:
    - Low: 0.0 <= breach_ratio < 0.5
    - Medium: 0.5 <= breach_ratio < 1.5
    - High: breach_ratio >= 1.5
    """
    if breach_ratio < 0.5:
        return "Low"
    elif breach_ratio < 1.5:
        return "Medium"
    else:
        return "High"


def classify_correlation_severity(abs_r: float, base_threshold: float = 0.70) -> str:
    """
    Classifies correlation severity based on Pearson coefficient strength:
    - Low: base_threshold <= |r| < base_threshold + 0.10
    - Medium: base_threshold + 0.10 <= |r| < 0.90
    - High: |r| >= 0.90
    """
    if abs_r < base_threshold:
        return "Low"

    if abs_r >= 0.90:
        return "High"
    elif abs_r >= (base_threshold + 0.10):
        return "Medium"
    else:
        return "Low"


def classify_threshold_breach_severity(current_val: float, benchmark_val: float, is_lower_breach: bool = True) -> str:
    """
    Classifies severity of an absolute benchmark threshold breach (e.g. Immunization < 80%).
    """
    gap = (benchmark_val - current_val) if is_lower_breach else (current_val - benchmark_val)
    if gap <= 5.0:
        return "Low"
    elif gap <= 15.0:
        return "Medium"
    else:
        return "High"
