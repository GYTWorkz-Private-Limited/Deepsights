# query_executor.py
import psycopg2
from .config import DATABASE_CONFIG

def execute_sql_query(sql_query):
    conn = None
    try:
        conn = psycopg2.connect(
            host=DATABASE_CONFIG["host"],
            port=DATABASE_CONFIG["port"],
            dbname=DATABASE_CONFIG["database"],
            user=DATABASE_CONFIG["user"],
            password=DATABASE_CONFIG["password"]
        )
        cursor = conn.cursor()
        cursor.execute(sql_query)
        result = cursor.fetchall()
        return result

    except Exception as e:
        return {"error": str(e)}

    finally:
        if conn:
            conn.close()
            
            
            

def get_unique_values_for_columns():
    # List of categorical columns to fetch unique values for
    categorical_columns = ['Customer', 'Vendor', 'Month', 'Transaction Type', 'Account', 'PNL Type']
    
    unique_values = {}
    
    for column in categorical_columns:
        # Dynamically create SQL query to fetch unique values for each column
        sql_query = f'SELECT DISTINCT "{column}" FROM vw_ai_rpt_pnl WHERE "{column}" IS NOT NULL;'
        
        # Execute the query to fetch unique values
        result = execute_sql_query(sql_query)
        
        # Store the result in a dictionary with the column name as the key
        if isinstance(result, list):
            unique_values[column] = [row[0] for row in result]  # Extract the unique values from the query result
        else:
            unique_values[column] = []  # In case of an error, keep it empty
    
    return unique_values