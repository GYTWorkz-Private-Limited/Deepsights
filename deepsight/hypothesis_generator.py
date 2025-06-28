import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from .query_executor import execute_sql_query
from .config import OPENAI_CONFIG, SCHEMA_SUMMARY

# Load environment variables
load_dotenv()

# Initialize OpenAI client with config
client = OpenAI(api_key=OPENAI_CONFIG['api_key'])

# Function to get unique values for each column by querying the database
def get_unique_values_for_columns():
    # List of categorical columns to fetch unique values for
    categorical_columns = ['Account Sub Type','Customer', 'Vendor', 'Month', 'Transaction Type', 'Account', 'PNL Type']
    
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

def generate_hypotheses(row, kpi_name, schema_summary):
    try:
        sample_name = row['Customer'] if pd.notna(row['Customer']) else row['Vendor']
        sample_name = str(sample_name).split('_')[0]  # Clean up any _C/_V suffixes
    except Exception as e:
        sample_name = "BigCorp"
        
    # Get unique values for categorical columns
    unique_values = get_unique_values_for_columns()
    
    # Format the unique values for each column as a string for easy reference in the prompt
    formatted_unique_values = "\n".join([f"- {col}: {', '.join(map(str, values))}" 
                                        for col, values in unique_values.items()])
    
    prompt = f"""
You are a financial analyst tasked with generating hypotheses for a detected anomaly in financial data. 
You must adhere strictly to the following instructions to ensure accurate and syntactically correct output:

KPI: {kpi_name}
Date: {row['Month']}
Value: {row[kpi_name]}
Change: {row['pct_change']:.2%} compared to rolling average

Schema Summary:
{schema_summary}

Unique values of categorical columns:
{formatted_unique_values}

Generate exactly 2 plausible hypotheses for this anomaly.
For each explanation, provide a corresponding SQL query to validate or disprove it.

Important Instructions:
- Always use correct column names: Customer, Vendor, Month, Revenue, Expense
- Do NOT use placeholder names like 'CustomerName_C' or 'VendorName_V'
- Do NOT use column aliases (e.g., Total_Revenue) in GROUP BY or HAVING clauses
- Unique Values: Use only the unique values provided above for categorical columns (e.g., for Transaction Type, use only 'Revenue' or 'Expense').
- No Extra Values: Do not introduce values outside the provided unique values (e.g., do not use 'Payment' for Transaction Type).
- Use realistic values like '{sample_name}' where applicable
- Wrap identifiers in double quotes if needed by PostgreSQL
- Format SQL inside triple backticks like:
```sql
SELECT ...
```
- Only include one SQL query per hypothesis
- Place SQL directly under 'Query:' without extra lines
- Avoid putting explanations inside SQL queries
- Always use the correct view name: vw_ai_rpt_pnl
- Start each query with SELECT
- Avoid extra text before or after SQL
- Start each SQL query with SELECT, INSERT, UPDATE, or DELETE
- Do NOT include comments like "-- This query will..."
- Avoid using column aliases unless in SELECT clause
- Use subqueries or CTEs if needed for multi-step logic
- Restrict Transaction Type values to 'Revenue' or 'Expense' only, as these are the only valid values
- Use unique values from the provided categorical columns where applicable
- Ensure that the SQL queries generated are syntactically correct
- Avoid errors like "syntax error at or near" by ensuring correct SQL syntax, especially for subqueries, HAVING clauses, or aggregates
- Avoid errors with functions like COUNT(), AVG(), etc., especially in the HAVING clause
- Do not use window functions in the HAVING clause
"""
    # Call the OpenAI API to generate hypotheses
    response = client.chat.completions.create(
        model=OPENAI_CONFIG['model'],
        messages=[{"role": "user", "content": prompt}],
        temperature=OPENAI_CONFIG['temperature'],
        max_tokens=OPENAI_CONFIG['max_tokens']
    )

    return response.choices[0].message.content