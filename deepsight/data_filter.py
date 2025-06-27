# data_filter.py
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
        return None, None

    finally:
        if conn:
            conn.close()

def process_data(df):
    # Fill NaNs and ensure string-safe fields
    df = df.copy()
    # Ensure numeric types
    df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce')
    df['Expense'] = pd.to_numeric(df['Expense'], errors='coerce')

    # Convert Date column to datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # Create Month column for grouping
        df['Month'] = df['Date'].dt.to_period('M').dt.to_timestamp()



    # Define common groupby columns
    group_cols_c = ['Month', 'Customer']
    group_cols_v = ['Month', 'Vendor' ]

    # Split and aggregate
    revenue_df = df[df['Customer'].notna()]
    
    expense_df = df[df['Vendor'].notna()]
    

    # Aggregate Revenue-side data by month
    if not revenue_df.empty:
        revenue_df = revenue_df.groupby(group_cols_c, dropna=False).agg({
            'Revenue': 'sum',
        }).reset_index()

    # Aggregate Expense-side data by month
    if not expense_df.empty:
        expense_df = expense_df.groupby(group_cols_v, dropna=False).agg({
            'Expense': 'sum'
        }).reset_index()
        
    # revenue_df.drop(columns=['Vendor'], inplace=True)
    # expense_df.drop(columns=['Customer'], inplace=True)

    return revenue_df, expense_df

# revenue_df, expense_df = fetch_data_for_realm("999999999")
# # print(revenue_df.columns)
# if revenue_df is not None:
#     print("Revenue Data Sample:")
#     print(revenue_df.head())
#     print(revenue_df.shape)

# if expense_df is not None:
#     print("\nExpense Data Sample:")
#     print(expense_df.head())
#     print(expense_df.shape)
    
    