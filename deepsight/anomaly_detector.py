import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest
from .config import ML_CONFIG

# Load environment variables
load_dotenv()

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
        rolling_window = ML_CONFIG['rolling_window']
        if group_col:
            # Group by Account Sub Type or Account for rolling calculations
            df['rolling_avg'] = df.groupby(group_col)[kpi_col].rolling(window=rolling_window).mean().shift(1).reset_index(level=0, drop=True)
        else:
            df['rolling_avg'] = df[kpi_col].rolling(window=rolling_window).mean().shift(1)
        
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

    # --- Method 2: Grouped Multivariate Anomaly Detection (Isolation Forest) ---
    # Apply ML anomaly detection within each group (similar to rule-based approach)

    if group_col:
        # Group-based ML anomaly detection (like rule-based)
        ml_based_anomalies_list = []

        for group_name, group_data in df.groupby(group_col):
            if len(group_data) < 3:  # Need minimum data for ML
                print(f"Warning: Insufficient data for ML anomaly detection in {group_col} '{group_name}' ({len(group_data)} rows)")
                continue

            # Prepare features for this specific group
            group_ml_anomalies = detect_ml_anomalies_for_group(group_data, kpi_col, group_name, group_col)
            if not group_ml_anomalies.empty:
                ml_based_anomalies_list.append(group_ml_anomalies)

        # Combine all group-based ML anomalies
        if ml_based_anomalies_list:
            ml_based_anomalies = pd.concat(ml_based_anomalies_list, ignore_index=True)
        else:
            ml_based_anomalies = pd.DataFrame()

        # Add anomaly columns to main dataframe
        df['anomaly_score'] = 1  # Default: not anomaly
        df['is_ml_anomaly'] = False

        if not ml_based_anomalies.empty:
            # Mark anomalies in main dataframe
            for _, anomaly_row in ml_based_anomalies.iterrows():
                mask = (df['Month'] == anomaly_row['Month']) & (df[group_col] == anomaly_row[group_col])
                df.loc[mask, 'anomaly_score'] = -1
                df.loc[mask, 'is_ml_anomaly'] = True
    else:
        # Fallback to original global ML approach for non-grouped data
        ml_based_anomalies = detect_ml_anomalies_global(df, kpi_col)
    
    return {
        'full_df': df,
        'rule_based_anomalies': rule_based_anomalies,
        'ml_based_anomalies': ml_based_anomalies
    }


def detect_ml_anomalies_for_group(group_data, kpi_col, group_name, group_col):
    """
    Detect ML anomalies within a specific group (Account Sub Type or Account)
    Similar to how rule-based detection works within groups
    """
    # Prepare features for this specific group's time series
    group_data = group_data.sort_values('Month').copy()

    # Create time series features for this group
    features_df = create_time_series_features(group_data, kpi_col)

    # Need at least 4 data points for meaningful ML analysis
    if len(features_df) < 4:
        return pd.DataFrame()

    # Select numeric features for ML
    feature_cols = [col for col in features_df.columns if col not in ['Month', group_col]]
    numeric_cols = features_df[feature_cols].select_dtypes(include=['number']).columns.tolist()

    if not numeric_cols:
        return pd.DataFrame()

    X = features_df[numeric_cols].copy()

    # Handle missing values and outliers
    X = preprocess_ml_features(X)

    if X.empty or len(X) < 3:
        return pd.DataFrame()

    # Apply Isolation Forest with higher contamination for group-level analysis
    contamination = min(0.2, max(0.1, 2.0 / len(X)))  # Adaptive contamination
    random_state = ML_CONFIG['random_state']
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state, n_estimators=50)

    try:
        anomaly_scores = iso_forest.fit_predict(X)

        # Get anomalies
        anomaly_mask = anomaly_scores == -1
        group_anomalies = features_df[anomaly_mask].copy()

        if not group_anomalies.empty:
            group_anomalies['anomaly_score'] = -1
            group_anomalies['is_ml_anomaly'] = True
            print(f"ML detected {len(group_anomalies)} anomalies in {group_col} '{group_name}'")

        return group_anomalies

    except Exception as e:
        print(f"Error in ML anomaly detection for {group_col} '{group_name}': {e}")
        return pd.DataFrame()


def create_time_series_features(group_data, kpi_col):
    """
    Create time series features for a specific group (Account Sub Type or Account)
    """
    features_df = group_data.copy()

    # Lag features (previous months)
    features_df[f'{kpi_col}_lag1'] = features_df[kpi_col].shift(1)
    features_df[f'{kpi_col}_lag2'] = features_df[kpi_col].shift(2)
    features_df[f'{kpi_col}_lag3'] = features_df[kpi_col].shift(3)

    # Rolling statistics (within this group)
    features_df[f'{kpi_col}_rolling_mean_3'] = features_df[kpi_col].rolling(window=3, min_periods=1).mean()
    features_df[f'{kpi_col}_rolling_std_3'] = features_df[kpi_col].rolling(window=3, min_periods=1).std()
    features_df[f'{kpi_col}_rolling_min_3'] = features_df[kpi_col].rolling(window=3, min_periods=1).min()
    features_df[f'{kpi_col}_rolling_max_3'] = features_df[kpi_col].rolling(window=3, min_periods=1).max()

    # Percentage change features
    features_df[f'{kpi_col}_pct_change'] = features_df[kpi_col].pct_change()
    features_df[f'{kpi_col}_pct_change_lag1'] = features_df[f'{kpi_col}_pct_change'].shift(1)

    # Trend features
    features_df[f'{kpi_col}_diff'] = features_df[kpi_col].diff()
    features_df[f'{kpi_col}_diff_lag1'] = features_df[f'{kpi_col}_diff'].shift(1)

    # Relative position features
    features_df[f'{kpi_col}_rank'] = features_df[kpi_col].rank(pct=True)
    features_df[f'{kpi_col}_zscore'] = (features_df[kpi_col] - features_df[kpi_col].mean()) / features_df[kpi_col].std()

    # Seasonal features (if applicable)
    if 'Month' in features_df.columns:
        features_df['month_num'] = pd.to_datetime(features_df['Month']).dt.month
        features_df['quarter'] = pd.to_datetime(features_df['Month']).dt.quarter
        features_df['is_year_end'] = (features_df['month_num'] == 12).astype(int)
        features_df['is_quarter_end'] = features_df['month_num'].isin([3, 6, 9, 12]).astype(int)

    return features_df


def preprocess_ml_features(X):
    """
    Preprocess features for ML anomaly detection
    """
    # Ensure numeric types
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    # Replace inf/-inf with NaN
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill NaN with median for each column
    for col in X.columns:
        if X[col].isna().any():
            median_val = X[col].median()
            if pd.isna(median_val):
                X[col].fillna(0, inplace=True)
            else:
                X[col].fillna(median_val, inplace=True)

    # Remove columns with zero variance
    for col in X.columns:
        if X[col].std() == 0:
            X = X.drop(columns=[col])

    # Cap extreme values at 99th percentile
    for col in X.columns:
        if len(X[col].unique()) > 1:  # Only if there's variation
            upper_bound = X[col].quantile(0.99)
            lower_bound = X[col].quantile(0.01)
            X[col] = X[col].clip(lower=lower_bound, upper=upper_bound)

    return X


def detect_ml_anomalies_global(df, kpi_col):
    """
    Fallback global ML anomaly detection (original approach)
    """
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if kpi_col not in numeric_cols and pd.api.types.is_numeric_dtype(df[kpi_col]):
        numeric_cols.append(kpi_col)

    if not numeric_cols:
        print("Error: No numeric columns available for Isolation Forest")
        return pd.DataFrame()

    X = df[numeric_cols].copy()
    X = preprocess_ml_features(X)

    if X.empty:
        print("Error: No valid data available for Isolation Forest after preprocessing")
        return pd.DataFrame()

    contamination = ML_CONFIG['contamination']
    random_state = ML_CONFIG['random_state']
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    df.loc[X.index, 'anomaly_score'] = iso_forest.fit_predict(X)
    df['is_ml_anomaly'] = df['anomaly_score'] == -1

    return df[df['is_ml_anomaly']]