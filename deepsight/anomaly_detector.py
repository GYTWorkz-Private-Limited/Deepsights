# anomaly_detector.py
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


def detect_monthly_kpi_anomalies(df, kpi_col="Column_name", threshold=1.0):
    """
    Detect anomalies in monthly aggregated KPIs.
    1. Compares current month vs average of previous 3 months (rule-based)
    2. Uses Isolation Forest for multivariate anomaly detection
    """

    df = df.sort_values('Month').copy().reset_index(drop=True)

    # --- Method 1: Rule-Based Anomaly Detection ---
    df['rolling_avg'] = df[kpi_col].rolling(window=3).mean().shift(1)
    df['pct_change'] = (df[kpi_col] - df['rolling_avg']) / df['rolling_avg'].abs()
    # min_base_value = 1000  # Only consider KPI values above this
    # df['is_rule_anomaly'] = (df['pct_change'].abs() > threshold) & (df[kpi_col] > min_base_value)
    df['is_rule_anomaly'] = df['pct_change'].abs() > threshold
    rule_based_anomalies = df[df['is_rule_anomaly']]


    # --- Method 2: Multivariate Anomaly Detection (Isolation Forest) ---
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if kpi_col not in numeric_cols:
        numeric_cols.append(kpi_col)

    X = df[numeric_cols].copy()

    #  Replace inf/-inf with NaN and fill with median
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(numeric_only=True), inplace=True)

    #  Optional: Cap extreme values at 99th percentile
    for col in X.columns:
        upper_bound = X[col].quantile(0.99)
        lower_bound = X[col].quantile(0.01)
        X[col] = X[col].clip(lower=lower_bound, upper=upper_bound)

    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    df['anomaly_score'] = iso_forest.fit_predict(X)
    df['is_ml_anomaly'] = df['anomaly_score'] == -1

    ml_based_anomalies = df[df['is_ml_anomaly']]
    
    
    return {
        'full_df': df,
        'rule_based_anomalies': rule_based_anomalies,
        'ml_based_anomalies': ml_based_anomalies
    }
    