import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def detect_monthly_kpi_anomalies(df, kpi_col="Column_name"):
    """
    Detect anomalies in monthly aggregated KPIs.
    1. Compares current month vs average of previous 3 months using IQR and probability-based anomaly detection
    2. Uses Isolation Forest for multivariate anomaly detection with robust null and infinite value handling
    """
    if df.empty or kpi_col not in df.columns:
        print(f"Error: Input DataFrame is empty or missing kpi_col '{kpi_col}'")
        return {
            'full_df': df,
            'rule_based_anomalies': pd.DataFrame(),
            'ml_based_anomalies': pd.DataFrame()
        }

    # Check if DataFrame is for Account Sub Type or Account
    is_account_sub_type = 'Account Sub Type' in df.columns
    is_account = 'Account' in df.columns
    group_col = 'Account Sub Type' if is_account_sub_type else 'Account' if is_account else None

    df = df.sort_values('Month').copy().reset_index(drop=True)

    # --- Method 1: Rule-Based Anomaly Detection (IQR + Probability-based) ---
    # Ensure sufficient data for rolling average
    if len(df) < 3 or (group_col and df.groupby(group_col).size().min() < 3):
        print(f"Warning: Insufficient data ({len(df)} rows or <3 rows per {group_col}) for rolling average. Skipping rule-based anomaly detection.")
        df['rolling_avg'] = np.nan
        df['pct_change'] = np.nan
        df['is_iqr_anomaly'] = False
        df['z_score'] = np.nan
        df['is_prob_anomaly'] = False
        df['is_rule_anomaly'] = False
        rule_based_anomalies = pd.DataFrame()
    else:
        # Calculate rolling average and percentage change
        if group_col:
            # Group by Account Sub Type or Account for rolling calculations
            df['rolling_avg'] = df.groupby(group_col)[kpi_col].rolling(window=3).mean().shift(1).reset_index(level=0, drop=True)
        else:
            df['rolling_avg'] = df[kpi_col].rolling(window=3).mean().shift(1)
        
        # Calculate pct_change, avoiding division by zero
        df['pct_change'] = np.where(
            df['rolling_avg'].abs() > 1e-10,  # Avoid division by zero
            (df[kpi_col] - df['rolling_avg']) / df['rolling_avg'].abs(),
            1000  # Replace inf with large finite value
        )
        df['pct_change'] = df['pct_change'].replace([np.inf, -np.inf], 1000)  # Handle any remaining inf

        # IQR-based anomaly detection
        Q1 = df['pct_change'].quantile(0.25)
        Q3 = df['pct_change'].quantile(0.75)
        IQR = Q3 - Q1
        iqr_lower_bound = Q1 - 1.5 * IQR
        iqr_upper_bound = Q3 + 1.5 * IQR
        df['is_iqr_anomaly'] = (df['pct_change'] < iqr_lower_bound) | (df['pct_change'] > iqr_upper_bound)
        
        # Probability-based anomaly detection (z-score method)
        pct_change_mean = df['pct_change'].mean()
        pct_change_std = df['pct_change'].std()
        if pd.isna(pct_change_std) or pct_change_std == 0:
            print("Warning: Cannot compute z_score (pct_change_std is NaN or 0). Skipping probability-based anomaly detection.")
            df['z_score'] = np.nan
            df['is_prob_anomaly'] = False
        else:
            df['z_score'] = (df['pct_change'] - pct_change_mean) / pct_change_std
            z_score_threshold = 1.96  # Corresponds to p-value < 0.05 (two-tailed)
            df['is_prob_anomaly'] = df['z_score'].abs() > z_score_threshold
        
        # Combine IQR and probability-based anomalies
        df['is_rule_anomaly'] = df['is_iqr_anomaly'] | df['is_prob_anomaly']
        rule_based_anomalies = df[df['is_rule_anomaly']]

    # --- Method 2: Multivariate Anomaly Detection (Isolation Forest) ---
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if kpi_col not in numeric_cols and pd.api.types.is_numeric_dtype(df[kpi_col]):
        numeric_cols.append(kpi_col)

    if not numeric_cols:
        print("Error: No numeric columns available for Isolation Forest")
        return {
            'full_df': df,
            'rule_based_anomalies': rule_based_anomalies,
            'ml_based_anomalies': pd.DataFrame()
        }

    X = df[numeric_cols].copy()

    # Ensure numeric types for all columns
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    # Replace inf/-inf with NaN
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill NaN with median for numeric columns
    X.fillna(X.median(numeric_only=True), inplace=True)

    # Check for remaining null values
    if X.isna().any().any():
        print(f"Warning: Null values detected in columns: {X.columns[X.isna().any()].tolist()}")
        X.dropna(inplace=True)
        df = df.loc[X.index].copy()

    # If X is empty after dropping nulls, return early
    if X.empty:
        print("Error: No valid data available for Isolation Forest after null handling")
        return {
            'full_df': df,
            'rule_based_anomalies': rule_based_anomalies,
            'ml_based_anomalies': pd.DataFrame()
        }

    # Optional: Cap extreme values at 99th percentile
    for col in X.columns:
        upper_bound = X[col].quantile(0.99)
        lower_bound = X[col].quantile(0.01)
        X[col] = X[col].clip(lower=lower_bound, upper=upper_bound)

    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    df.loc[X.index, 'anomaly_score'] = iso_forest.fit_predict(X)
    df['is_ml_anomaly'] = df['anomaly_score'] == -1

    ml_based_anomalies = df[df['is_ml_anomaly']]
    
    return {
        'full_df': df,
        'rule_based_anomalies': rule_based_anomalies,
        'ml_based_anomalies': ml_based_anomalies
    }