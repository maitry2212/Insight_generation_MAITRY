# Software Requirements Specification (SRS)
## Project: Automated Insight Generation — AI/ML

**Document Version:** 1.0.0  
**Status:** Draft / Approved for Implementation  
**Target Platform:** Python 3.10+ / Streamlit / Pandas / NumPy / Plotly  

---

## 1. Executive Summary & Problem Statement

### 1.1 Problem Statement
Public health administrators and district health officers track multiple health indicators across administrative units (e.g., Antenatal Care [ANC] coverage, institutional delivery rates, immunization rates, and high-risk pregnancy counts). However, raw tabular records make it difficult to quickly spot:
- Sharp month-over-month performance deterioration or surges (Trends).
- Anomalous district readings deviating significantly from state norms or peer distributions (Outliers).
- Inter-indicator associations and structural anomalies (e.g., rising high-risk cases accompanied by declining ANC coverage) (Correlations).

Traditional manual reporting is slow, error-prone, and lacks consistent mathematical rigor. Furthermore, static narrative generation frequently suffers from hardcoded, non-scalable rules.

### 1.2 Proposed Solution
The **Automated Insight Generation — AI/ML** application is a lightweight, interactive auto-analytics engine that:
1. Ingests district-level healthcare CSV datasets.
2. Performs automated data validation, profiling, and missing value diagnostics.
3. Automatically computes mathematical trends, statistical outliers (IQR and Z-Score), and Pearson correlation matrices.
4. Generates standardized, structured, human-readable insight objects parameterized entirely by data.
5. Dynamically assigns severity levels (Low, Medium, High) using mathematically derived, configurable thresholds.
6. Presents findings in an interactive Streamlit dashboard featuring live filters, severity distribution charts, correlation heatmaps, district time-series plots, and exportable CSV reports.

---

## 2. Project Objectives & Scope

### 2.1 Project Objectives
- **Automated Mathematical Diagnostics:** Eliminate manual data scanning by automatically running trend, outlier, and correlation engines.
- **Dynamic Templated Storytelling:** Transform statistical anomalies into standardized natural language sentences without hardcoding district names or static threshold values.
- **Configurability:** Enable health domain experts to tune sensitivity thresholds (e.g., trend % change threshold, Z-score cutoff, Pearson $|r|$ cutoff) in real-time via UI controls.
- **Actionable Decision Support:** Categorize every detected anomaly into Low, Medium, or High severity to guide priority field interventions.
- **Transparency & Integrity:** Clearly surface data sample size limitations (e.g., low sample size warnings when computing correlations on small datasets).

### 2.2 In-Scope
- Ingestion of standard CSV files containing monthly district healthcare metrics.
- Data validation, schema verification, data type conversion, and missing-value profiling.
- Core Statistical Engines:
  - **Trend Analysis:** Month-over-month (MoM) percentage change per district and indicator.
  - **Outlier Detection:** Dual-engine support for Interquartile Range (IQR) method and Parametric Z-Score method.
  - **Correlation Detection:** Pearson product-moment correlation matrix computation across all numerical indicator pairs.
- Automated Structured Insight Generation with 9 mandatory standardized fields (`insight_id`, `type`, `indicator`, `entity`, `period`, `metric`, `change`, `severity`, `explanation`).
- Dynamic Data-Driven Severity Classifier.
- Interactive Dashboard UI (Streamlit) with reactive filtering (District, Month, Indicator), Plotly charts (bar, heatmap, multi-district trend line chart), and tabular summaries.
- Export functionality for generated insights CSV and Pearson correlation matrix CSV.

### 2.3 Out-of-Scope (Constraints & Non-Goals)
- Multi-tier client-server architecture or separate REST API backend (Single-tier Streamlit Python runtime).
- Write-back to external transactional databases (EHR/EMR systems).
- Machine learning deep neural network models or external black-box LLM API calls (System must rely on deterministic, explainable, and reproducible statistical algorithms).
- Authentication, RBAC (Role-Based Access Control), or multi-tenant database partitioning.

---

## 3. Input Dataset Schema & Data Validation

### 3.1 Input Schema Specification

| Field Name | Type | Unit / Format | Required | Valid Range / Constraints |
| :--- | :--- | :--- | :--- | :--- |
| `month` | String / Date | `YYYY-MM` or `YYYY-MM-DD` | Yes | Valid ISO calendar date format |
| `district` | String | Categorical string | Yes | Non-empty alphanumeric string |
| `anc_coverage` | Float / Int | Percentage ($0 - 100\%$) | Yes | Continuous numerical metric $\ge 0$ |
| `institutional_delivery` | Float / Int | Percentage ($0 - 100\%$) | Yes | Continuous numerical metric $\ge 0$ |
| `immunization` | Float / Int | Percentage ($0 - 100\%$) | Yes | Continuous numerical metric $\ge 0$ |
| `high_risk_cases` | Integer | Raw incident count | Yes | Discrete non-negative integer $\ge 0$ |

### 3.2 Granularity & Data Assumptions
- **Granularity:** Exactly 1 row per `(district, month)` tuple.
- **Minimum Data Requirement:** $\ge 2$ consecutive months for trend computation; $\ge 3$ months recommended for trend stability; $\ge 10$ districts recommended for stable correlation estimation.

### 3.3 Data Ingestion & Missing Value Handling Rules
- **Validation Check 1 (Schema Compliance):** Check presence of all mandatory columns. If any required column is missing, abort processing and display explicit UI error.
- **Validation Check 2 (Duplicate Tuples):** Check for duplicate `(district, month)` rows. If found, highlight duplicates and offer user-selected aggregation (mean/last) or rejection.
- **Validation Check 3 (Missing Values):** 
  - Compute and display count and percentage of missing values per column via `df.isna().sum()`.
  - Provide configurable handling strategies: (a) Drop rows with missing metric values, (b) Forward-fill per district time series, or (c) Impute with district/state median.
- **Validation Check 4 (Type Casting & Cleaning):** Sanitize string whitespace, cast `month` to sortable datetime object, and cast numerical indicators to `float64`/`int64`.

---

## 4. Analytics & Algorithm Engine Specifications

### 4.1 Trend Detection Engine (Part B)
- **Algorithm:** For each distinct `(district, indicator)` time-series sorted chronologically by `month`:
  $$\Delta\% = \left(\frac{V_{t} - V_{t-1}}{V_{t-1}}\right) \times 100$$
  *(Where $V_t$ is the metric value at the current month and $V_{t-1}$ is the metric value at the immediate preceding month).*
- **Edge Case (Zero Base Value):** If $V_{t-1} = 0$:
  - If $V_t = 0 \implies \Delta\% = 0\%$.
  - If $V_t > 0 \implies \Delta\% = +\infty$ (Handled gracefully as absolute increment: $+V_t$ units).
- **Flagging Rule:** An event is flagged as significant if:
  $$|\Delta\%| \ge \theta_{\text{trend}}$$
  *Default configurable threshold:* $\theta_{\text{trend}} = 10.0\%$ (Adjustable via UI slider $1\% - 50\%$).

---

### 4.2 Outlier Detection Engine (Part C)
The system supports two complementary outlier detection strategies across all districts within each month (or state-wide cross-sectional pool):

#### Option 1: Interquartile Range (IQR) Method (Non-parametric)
- Let $Q_1 = \text{25th percentile}$, $Q_3 = \text{75th percentile}$, and $\text{IQR} = Q_3 - Q_1$.
- Lower Bound: $LB = Q_1 - (k \times \text{IQR})$
- Upper Bound: $UB = Q_3 + (k \times \text{IQR})$
- Default multiplier: $k = 1.5$ (Standard Tukey fence)
- Outlier condition: $V < LB \quad \text{OR} \quad V > UB$.

#### Option 2: Z-Score Method (Parametric)
- Let $\mu = \text{mean}(V)$ and $\sigma = \text{standard\_deviation}(V)$ for the indicator within the period.
- Standardized Score:
  $$Z = \frac{V - \mu}{\sigma}$$
- Outlier condition: $|Z| \ge \theta_Z$
- Default configurable threshold: $\theta_Z = 3.0$ (Adjustable via UI slider $1.5 - 4.0$).
- *Note:* If $\sigma = 0$ (all values identical), $Z = 0$ (no outlier).

---

### 4.3 Pearson Correlation Detection Engine (Part D)
- **Algorithm:** For every distinct pair of numerical indicators $(X, Y)$, compute the Pearson product-moment correlation coefficient $r_{XY}$:
  $$r_{XY} = \frac{\sum (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum (X_i - \bar{X})^2 \sum (Y_i - \bar{Y})^2}}$$
- **Correlation Matrix:** Generate a full symmetric matrix for all numerical indicators.
- **Flagging Rule:** A pair $(X, Y)$ is flagged as strongly correlated if:
  $$|r_{XY}| \ge \theta_{\text{corr}}$$
  *Default configurable threshold:* $\theta_{\text{corr}} = 0.70$ (Adjustable via UI slider $0.30 - 0.95$).
- **Sample Size Fragility Disclaimer:**
  When $N < 30$ (e.g., 6 districts $\times$ 2 months = 12 data points), the UI must explicitly display a statistical fragility warning banner informing the user that correlation coefficients have wider confidence intervals and may reflect small-sample volatility.

---

### 4.4 Automated Insight Generation Specification (Part E)

Each generated anomaly/finding is translated into a standardized insight record.

#### Mandatory Structured Output Schema

| Field Name | Type | Description | Example Values |
| :--- | :--- | :--- | :--- |
| `insight_id` | String | Unique sequential identifier (`INS-XXXX`) | `INS-0001`, `INS-0002` |
| `type` | String | Categorical insight class | `trend`, `outlier`, `correlation`, `threshold_breach` |
| `indicator` | String | Healthcare metric(s) analyzed | `anc_coverage`, `high_risk_cases` |
| `entity` | String | Administrative district or State | `Ahmedabad`, `Mehsana`, `State-wide` |
| `period` | String | Relevant time period or interval | `2026-08`, `2026-07..2026-08` |
| `metric` | Float / String | Current measured value or coefficient | `69.0`, `42.0`, `r = -0.98` |
| `change` | Float / String | Relative delta, Z-score, or shift | `-18.8%`, `-3.1 sigma`, `+17 cases` |
| `severity` | String | Data-driven importance level | `Low`, `Medium`, `High` |
| `explanation` | String | Human-readable templated sentence | *"ANC Coverage in Ahmedabad dropped by 18.8% compared to the previous month, exceeding the 10.0% significant-change threshold."* |

#### Dynamic Templating Logic (No Hardcoded Narratives)
- **Trend Template:**  
  `"{indicator_name} in {entity} {direction} by {abs_change}% from {prev_value} to {current_value} in {period}, exceeding the {threshold}% significance threshold."`
- **Outlier Template (Z-Score):**  
  `"{entity}'s {indicator_name} of {current_value} in {period} is {z_score:.1f}σ {below_above} the group mean ({group_mean:.1f}), flagging as an anomalous statistical outlier."`
- **Outlier Template (IQR):**  
  `"{entity}'s {indicator_name} of {current_value} in {period} falls outside the normal IQR boundary [{lower_bound:.1f}, {upper_bound:.1f}]."`
- **Correlation Template:**  
  `"Strong {direction} correlation (r = {r_val:.2f}) observed between {indicator_1} and {indicator_2} across {sample_size} observations in {period}."`

---

## 5. Data-Driven Severity Classification Rules (Part 10)

Severity is never assigned via arbitrary magic constants. Instead, it is computed dynamically based on the degree of deviation relative to user-configured thresholds or statistical distribution tiers:

### 5.1 Trend Severity Matrix

| Relative Deviation Magnitude ($|\Delta\%|$) | Assigned Severity | Rationale |
| :--- | :--- | :--- |
| $\theta_{\text{trend}} \le |\Delta\%| < 1.5 \times \theta_{\text{trend}}$ (e.g. $10\% - 14.9\%$) | **Low** | Noticeable shift slightly above baseline threshold |
| $1.5 \times \theta_{\text{trend}} \le |\Delta\%| < 2.5 \times \theta_{\text{trend}}$ (e.g. $15\% - 24.9\%$) | **Medium** | Substantial operational shift requiring monitoring |
| $|\Delta\%| \ge 2.5 \times \theta_{\text{trend}}$ (e.g. $\ge 25\%$) | **High** | Critical performance swing requiring immediate intervention |

### 5.2 Outlier Severity Matrix

| Outlier Method | Low Severity | Medium Severity | High Severity |
| :--- | :--- | :--- | :--- |
| **Z-Score Method** | $\theta_Z \le \|Z\| < \theta_Z + 0.5$ | $\theta_Z + 0.5 \le \|Z\| < \theta_Z + 1.2$ | $\|Z\| \ge \theta_Z + 1.2$ (or $\|Z\| \ge 3.5$) |
| **IQR Method** | Mild outlier ($1.5 \times \text{IQR} \le \text{dist} < 2.0 \times \text{IQR}$) | Moderate ($2.0 \times \text{IQR} \le \text{dist} < 3.0 \times \text{IQR}$) | Extreme outlier ($\ge 3.0 \times \text{IQR}$) |

### 5.3 Correlation Severity Matrix

| Absolute Correlation ($|r|$) | Assigned Severity |
| :--- | :--- |
| $\theta_{\text{corr}} \le \|r\| < \theta_{\text{corr}} + 0.10$ | **Low** |
| $\theta_{\text{corr}} + 0.10 \le \|r\| < 0.90$ | **Medium** |
| $\|r\| \ge 0.90$ | **High** |

---

## 6. User Interface & Visualization Specifications

### 6.1 UI Layout Architecture (Streamlit)
1. **Sidebar Controls:**
   - CSV File Uploader with sample dataset fallback / reset button.
   - Threshold Adjustment Sliders:
     - Trend Significance Threshold ($\theta_{\text{trend}}$: $5\% - 50\%$, default: $10\%$).
     - Outlier Method Selector (`Z-Score` vs `IQR`) & Outlier Cutoff Slider ($\theta_Z$: $1.5 - 4.0$, default: $3.0$).
     - Pearson Correlation Cutoff Slider ($\theta_{\text{corr}}$: $0.50 - 0.95$, default: $0.70$).
   - Interactive Filters:
     - Multi-District Selector (Default: All).
     - Month Range / Selector (Default: All).
     - Indicator Selector (Default: All).
     - Severity Filter (`All`, `High`, `Medium`, `Low`).
2. **Main Dashboard View:**
   - **Section 1: Data Summary & Health Diagnostics:** KPI metric cards (Total Records, Districts, Date Range, Missing Values Count), Data Preview Table (`head()`, `info()`).
   - **Section 2: Automated Insights Feed:** Filterable structured insights table with color-coded severity badges (🔴 High, 🟡 Medium, 🔵 Low) and expandable explanatory cards.
   - **Section 3: Interactive Visualizations (Plotly):**
     - *Chart A:* Severity Distribution (Categorical Bar Chart).
     - *Chart B:* Pearson Correlation Matrix (Interactive Heatmap with annotated $r$ values).
     - *Chart C:* District Performance Time-Series (Multi-line chart with threshold and outlier markers).
     - *Chart D:* Outlier Box-Plots per indicator.
   - **Section 4: Data & Insights Export:** One-click download buttons for `insights.csv` and `correlation_matrix.csv`.

---

## 7. Non-Functional & Quality Requirements

- **NFR-01 (Performance & Latency):** Analysis and insight generation must execute in $< 1.0\text{ second}$ for datasets up to 10,000 rows.
- **NFR-02 (Determinism & Explainability):** Given identical inputs and slider configurations, the insight engine must generate exact, reproducible outputs without stochastic drift.
- **NFR-03 (Fault Tolerance & Graceful Degradation):** Malformed CSV headers, non-numeric strings in metric columns, or single-month files must produce clear diagnostic alerts rather than runtime unhandled exceptions.
- **NFR-04 (UI Usability & Responsiveness):** Clean, modern styling using Plotly interactive charts (hover tooltips, zoom, pan) and Streamlit container-based layout.
- **NFR-05 (Portability):** Zero native binary dependencies outside standard Python packages (`streamlit`, `pandas`, `numpy`, `plotly`, `scipy`).

---

## 8. Acceptance Criteria & Test Cases

| Req ID | Test Scenario | Input / Action | Expected Result | Pass / Fail Criteria |
| :--- | :--- | :--- | :--- | :--- |
| **AC-01** | Sample CSV Ingestion | Upload assignment CSV (6 districts $\times$ 2 months) | Correctly parses 12 rows, displays 0 missing values, and shows schema | Data table rendered with 6 columns |
| **AC-02** | MoM Trend Detection | Trend threshold = 10% | Flags Ahmedabad `anc_coverage` (85 $\to$ 69, $-18.8\%$) and Mehsana `anc_coverage` (84 $\to$ 42, $-50.0\%$) | Both trends captured with accurate $\Delta\%$ |
| **AC-03** | Outlier Detection | Mehsana `anc_coverage` = 42 in `2026-08` | Z-score or IQR identifies Mehsana as an outlier vs group mean (~76) | Anomaly flagged with severity HIGH |
| **AC-04** | Correlation Matrix | Compute Pearson correlation | Computes exact correlation matrix; flags $|r| \ge 0.70$ and displays low-sample disclaimer ($N=12$) | Heatmap rendered + disclaimer shown |
| **AC-05** | Structured Insight Format | Generate Insights Table | Inspect insight columns | All 9 required fields present (`insight_id` to `explanation`) |
| **AC-06** | Dynamic Severity | Threshold $= 10\%$ | Ahmedabad $-18.8\%$ classified based on data rules | Appropriate severity (Medium/High) without hardcoding |
| **AC-07** | CSV Export | Click "Download Insights CSV" | Downloads CSV matching required structure | File contains valid CSV headers and records |

---

## 9. Key Architectural Decisions & Ambiguities Identified

Before proceeding to full application implementation, the following design decisions and ambiguities have been analyzed:

1. **Date Format Flexibility:** The sample uses `2026-07` while standard datetime parsing expects full dates (`2026-07-01`). The engine should support both month-year (`YYYY-MM`) and day-accurate dates (`YYYY-MM-DD`).
2. **Outlier Grouping Level (Cross-Sectional vs Temporal):** In healthcare monitoring, an outlier can either be detected:
   - *Cross-sectionally per month* (e.g., comparing all 6 districts in August 2026).
   - *Longitudinally per district* (comparing a district against its own historical distribution across many months).  
   *Decision:* The system will default to cross-sectional per-month evaluation (ideal for small month counts like $M=2$), with an option for overall pool evaluation.
3. **Threshold Breach Insight Type:** The assignment schema lists `type` $\in$ `{trend, outlier, correlation, threshold_breach}`. We can allow optional absolute benchmark thresholds (e.g. Immunization $< 80\%$ critical health goal breach).
4. **Reactive State Management:** Streamlit UI sliders should trigger immediate recalculation of all insight engines and Plotly figures without requiring a full app reload.
