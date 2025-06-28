import psycopg2
import pandas as pd

from .config import DATABASE_CONFIG

def fetch_data_for_realm(realm_id: str):
    conn = None
    try:
        conn = psycopg2.connect(
            host=DATABASE_CONFIG["host"],
            port=DATABASE_CONFIG["port"],
            dbname=DATABASE_CONFIG["database"],
            user=DATABASE_CONFIG["user"],
            password=DATABASE_CONFIG["password"]
        )

        query = """
        SELECT *
        FROM vw_ai_rpt_pnl
        WHERE realm_id = %s
        """

        cursor = conn.cursor()
        cursor.execute(query, (realm_id,))
        result = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(result, columns=columns)
        cursor.close()
        return process_data(df)

    except Exception as e:
        print(f"Error fetching or processing data: {e}")
        return None, None, None, None, None, None  # Return None for all six DataFrames on error

    finally:
        if conn:
            conn.close()

def process_data(df):
    # Validate required columns
    required_columns = ['Customer', 'Vendor', 'Revenue', 'Expense', 'Date']
    optional_columns = ['Account Sub Type', 'Account']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns in DataFrame: {missing_columns}")
        return None, None, None, None, None, None

    # Fill NaNs and ensure string-safe fields
    df = df.copy()
    # Ensure numeric types
    df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce')
    df['Expense'] = pd.to_numeric(df['Expense'], errors='coerce')

    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Month'] = df['Date'].dt.to_period('M').dt.to_timestamp()

    # Define common groupby columns
    group_cols_c = ['Month', 'Customer']
    group_cols_v = ['Month', 'Vendor']
    group_cols_AS = ['Month', 'Account Sub Type'] if 'Account Sub Type' in df.columns else None
    group_cols_A = ['Month', 'Account'] if 'Account' in df.columns else None

    # Initialize DataFrames
    revenue_df = pd.DataFrame()
    expense_df = pd.DataFrame()
    AS_df_revenue = pd.DataFrame()
    AS_df_expense = pd.DataFrame()
    A_df_revenue = pd.DataFrame()
    A_df_expense = pd.DataFrame()

    # Aggregate Revenue-side data by month
    revenue_df_raw = df[df['Customer'].notna()]
    if not revenue_df_raw.empty:
        revenue_df = revenue_df_raw.groupby(group_cols_c, dropna=False).agg({
            'Revenue': 'sum',
        }).reset_index()

    # Aggregate Expense-side data by month
    expense_df_raw = df[df['Vendor'].notna()]
    if not expense_df_raw.empty:
        expense_df = expense_df_raw.groupby(group_cols_v, dropna=False).agg({
            'Expense': 'sum'
        }).reset_index()

    # Account Sub Type aggregation
    if group_cols_AS and 'Account Sub Type' in df.columns:
        AS_df_raw = df[df['Account Sub Type'].notna()]
        if not AS_df_raw.empty:
            # Revenue aggregation
            AS_df_revenue = AS_df_raw.groupby(group_cols_AS, dropna=False).agg({
                'Revenue': 'sum',
            }).reset_index()
            # Expense aggregation
            AS_df_expense = AS_df_raw.groupby(group_cols_AS, dropna=False).agg({
                'Expense': 'sum',
            }).reset_index()

    # Account aggregation
    if group_cols_A and 'Account' in df.columns:
        A_df_raw = df[df['Account'].notna()]
        if not A_df_raw.empty:
            # Revenue aggregation
            A_df_revenue = A_df_raw.groupby(group_cols_A, dropna=False).agg({
                'Revenue': 'sum',
            }).reset_index()
            # Expense aggregation
            A_df_expense = A_df_raw.groupby(group_cols_A, dropna=False).agg({
                'Expense': 'sum',
            }).reset_index()

    return revenue_df, expense_df, AS_df_revenue, AS_df_expense, A_df_revenue, A_df_expense

# revenue_df, expense_df, AS_df_revenue, AS_df_expense, A_df_revenue, A_df_expense = fetch_data_for_realm("999999999")

# if AS_df_revenue is not None:
#     print("\nAccount Sub Type Revenue Data Sample:")
#     print(AS_df_revenue.head())
#     print(AS_df_revenue.shape)

# if AS_df_expense is not None:
#     print("\nAccount Sub Type Expense Data Sample:")
#     print(AS_df_expense.head())
#     print(AS_df_expense.shape)

# if A_df_revenue is not None:
#     print("\nAccount Revenue Data Sample:")
#     print(A_df_revenue.head())
#     print(A_df_revenue.shape)

# if A_df_expense is not None:
#     print("\nAccount Expense Data Sample:")
#     print(A_df_expense.head())
#     print(A_df_expense.shape)

# if expense_df is not None:
#     print("\nExpense Data Sample:")
#     print(expense_df.head())
#     print(expense_df.shape)