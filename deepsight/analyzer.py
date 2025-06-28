import re
from datetime import datetime
from .anomaly_detector import detect_monthly_kpi_anomalies
from .hypothesis_generator import generate_hypotheses
from .query_executor import execute_sql_query
import os
import pandas as pd
from sqlparse import parse, tokens
from openai import OpenAI

def parse_hypotheses_and_queries_strict(raw_response):
    """
    Stricter parser that extracts only the SQL query within ```sql ... ``` blocks.
    """
    pattern = r"```sql\n(.*?)\n```"
    matches = re.findall(pattern, raw_response, re.DOTALL)
    
    results = []
    for i, query in enumerate(matches, 1):
        query = query.strip()
        if not query.endswith(';'):
            query += ';'
        results.append({
            "hypothesis_number": i,
            "hypothesis": f"Hypothesis {i} extracted from raw response",
            "query": query
        })
    
    return results

def parse_hypotheses_and_queries_original(raw_response):
    """
    Original parser that captures hypothesis and query with explanation.
    """
    pattern = r"(?:Explanation|Hypothesis)\s+(\d+):\s*(.*?)\s*Query:\s*((?:.|\n)*?)(?=(?:\n(?:Explanation|Hypothesis)\s+\d+:|\Z))"
    matches = re.findall(pattern, raw_response, re.DOTALL)

    results = []
    for hyp_num, hypothesis, query in matches:
        query = re.sub(r'```sql\n|```', '', query).strip()
        results.append({
            "hypothesis_number": int(hyp_num),
            "hypothesis": hypothesis.strip(),
            "query": query.strip()
        })

    return results

def is_valid_sql_query(query):
    """
    Validate if the query is syntactically correct SQL.
    """
    try:
        parsed = parse(query)
        if not parsed:
            return False
        query_upper = query.upper()
        sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH']
        has_sql_keyword = any(keyword in query_upper for keyword in sql_keywords)
        if 'EXPLANATION:' in query_upper:
            return False
        return has_sql_keyword
    except Exception:
        return False

def parse_hypotheses_and_queries(raw_response):
    """
    Combined parser that tries both strict and original parsers and validates results.
    """
    strict_results = parse_hypotheses_and_queries_strict(raw_response)
    valid_strict = all(is_valid_sql_query(item['query']) for item in strict_results)
    
    original_results = parse_hypotheses_and_queries_original(raw_response)
    valid_original = all(is_valid_sql_query(item['query']) for item in original_results)
    
    if valid_strict and strict_results:
        return strict_results
    elif valid_original and original_results:
        return original_results
    else:
        print("⚠️ Warning: No fully valid SQL queries found. Falling back to strict parser.")
        return strict_results if strict_results else original_results

def sanitize_sql_query(query):
    """
    Sanitize SQL query by replacing placeholder table/column names and cleaning up.
    """
    query = query.replace("your_table_name", "vw_ai_rpt_pnl") \
        .replace("financial_data", "vw_ai_rpt_pnl") \
        .replace("customer", "Customer") \
        .replace("vendor", "Vendor") \
        .replace("Month", "Month") \
        .replace("Revenue", "Revenue") \
        .replace("Expense", "Expense")
    query = re.sub(r'\n\s*Explanation:.*', '', query, flags=re.DOTALL)
    return query.strip()

def fix_group_by_error(query, error_message):
    """
    Attempt to fix SQL query errors related to GROUP BY by moving non-grouped columns
    from HAVING to WHERE clause.
    """
    if "must appear in the GROUP BY clause or be used in an aggregate function" in error_message:
        match = re.search(r'column "([^"]+)" must appear', error_message)
        if not match:
            return query
        
        problematic_column = match.group(1)
        column_name = problematic_column.split('.')[-1]
        
        having_pattern = r'(HAVING\s+)(.*?)(;|\Z)'
        having_match = re.search(having_pattern, query, re.IGNORECASE | re.DOTALL)
        if not having_match:
            return query
        
        having_clause = having_match.group(2)
        condition_pattern = rf'\b{column_name}\b\s*(=|>|<|>=|<=|!=)\s*\'[^\']+\''
        condition_match = re.search(condition_pattern, having_clause, re.IGNORECASE)
        if not condition_match:
            return query
        
        condition = condition_match.group(0)
        new_having = re.sub(rf'\b{condition}\b(\s*AND\s*)?', '', having_clause, flags=re.IGNORECASE).strip()
        if new_having and not new_having.lower().startswith('and'):
            new_having = f"AND {new_having}"
        new_having = new_having.strip()
        
        where_pattern = r'(WHERE\s+.*?)(GROUP\s+BY|HAVING|;|\Z)'
        where_match = re.search(where_pattern, query, re.IGNORECASE | re.DOTALL)
        if where_match:
            where_clause = where_match.group(1).strip()
            new_where = f"{where_clause} AND {condition}" if where_clause.lower() != 'where' else f"WHERE {condition}"
            new_query = re.sub(where_pattern, f"{new_where} \\2", query, flags=re.IGNORECASE | re.DOTALL)
        else:
            new_query = re.sub(r'(FROM\s+.*?)(GROUP\s+BY)', f'\\1 WHERE {condition} \\2', query, flags=re.IGNORECASE | re.DOTALL)
        
        if new_having and new_having != 'AND':
            new_query = re.sub(having_pattern, f'HAVING {new_having}\\3', new_query, flags=re.IGNORECASE | re.DOTALL)
        else:
            new_query = re.sub(r'\s*HAVING\s+.*?;', ';', new_query, flags=re.IGNORECASE | re.DOTALL)
        
        return new_query.strip()
    
    return query

SCHEMA_SUMMARY = """
Columns:
- Month: First day of the month (YYYY-MM-DD)
- Customer: Customer name (NULL if expense)
- Vendor: Vendor name (NULL if revenue)
- Account: General Ledger account name
- Account Sub Type: General Ledger account name    
- PNL Type: Budget, Forecast, Actual
- Transaction Type: Invoice, Payment, Refund, etc.
- Revenue: Total revenue for the month
- Expense: Total expense for the month
- Net Revenue: Revenue net of discounts/refunds
"""

def save_anomalies_to_csv(anomalies_df, entity_type, kpi_column, output_dir="anomalies_output"):
    """
    Save anomalies DataFrame to a CSV file with timestamped filename.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/anomalies_{entity_type}_{kpi_column}_{timestamp}.csv"
    anomalies_df.to_csv(filename, index=False)
    print(f"\n[+] Saved anomalies to: {filename}")

def save_markdown_report(content, entity_type, kpi_column, output_dir="output/reports"):
    """
    Save markdown report with timestamped filename.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/combined_anomalies_report_{entity_type}_{kpi_column}_{timestamp}.md"
    with open(filename, 'w') as f:
        f.write(content)
    print(f"\n[+] Saved Combined Anomalies Report to: {filename}")
    return filename

def generate_combined_insight(anomalies_data, kpi_column, entity_type, mindset="growth"):
    """
    Generate a combined insight report summarizing all anomalies in a business-friendly format.
    """
    mindset_guidance = {
        "growth": "Focus on opportunities for expansion and positive outcomes.",
        "risk": "Focus on potential threats and mitigation strategies.",
        "efficiency": "Focus on process improvements and cost savings."
    }
    
    mindset_prompt = mindset_guidance.get(mindset.lower(), mindset_guidance["growth"])
    
    # Aggregate total change and top contributors
    total_change = sum(anomaly['row'][kpi_column] * anomaly['row']['pct_change'] for anomaly in anomalies_data)
    total_value = sum(anomaly['row'][kpi_column] for anomaly in anomalies_data)
    change_percent = (total_change / total_value) * 100 if total_value else 0
    
    # Collect contributors from validation results where supported
    contributors = {}
    for anomaly in anomalies_data:
        for item in anomaly['validation_results']:
            if item['result_count'] > 0:
                amount_match = re.search(r'increased by \$(\d+)', item['hypothesis'])
                if amount_match:
                    amount = int(amount_match.group(1))
                    entity_match = re.search(r'(Account \d+|Vendor \d+|Customer \w+|Transaction Type \w+)', item['hypothesis'])
                    if entity_match:
                        entity = entity_match.group(0)
                        contributors[entity] = contributors.get(entity, 0) + amount
    
    # Sort contributors by amount
    sorted_contributors = sorted(contributors.items(), key=lambda x: x[1], reverse=True)
    
    prompt = f"""
You are a financial analyst tasked with summarizing financial anomalies in {entity_type} data using a {mindset} mindset. 
{mindset_prompt}

**Data Schema**:
{SCHEMA_SUMMARY}

**Summary Data**:
- Total {kpi_column} across anomalies: ${total_value:,.0f}k with a change of ${total_change:,.0f}k ({change_percent:.0f}% vs previous period)

**Task**:
Provide a concise business summary (200-300 words) of the anomalies based on the total value, change, and top contributors from the validation results with . List the main factors driving the change in descending order of impact, including their contribution amounts and percentage increases where available. Avoid technical terms like 'hypothesis' or 'SQL query.' or 'anomaly' Focus on business insights aligned with the {mindset} mindset. Format the response in markdown with:

1. **Overview**: State the total {kpi_column} and overall change andd also what it chnage with respect to some time period where we have seen change. 
2. **Key Contributors**: List the top factors contributing to the change with their percentage change in the total value also with with their compare verison witht he verison months ot time perdiod if their is some change seen in that facotr (if available).
3. **Insights**: Highlight business implications and one actionable recommendation.

Use the schema (e.g.,Account Sub Type, Account, Vendor, Customer, Transaction Type, PNL Type) and validation results to ground the summary, identifying contributors dynamically based on the data.
"""
    client = OpenAI(api_key="api_key")
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content

def run_deepsight_analysis(df, kpi_column, entity_type, mindset="growth"):
    """
    Run DeepSight analysis with a combined anomalies report in the desired format.
    """
    print(f"\n[+] Running DeepSight analysis on {entity_type} data with {mindset} mindset...")
    result = detect_monthly_kpi_anomalies(df, kpi_column)
    full_df = result['full_df']
    rule_anomalies = result['rule_based_anomalies']
    ml_anomalies = result['ml_based_anomalies']

    output_dir = "output/anomalies"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/flagged_data_{entity_type}_{kpi_column}_{timestamp}.csv"
    full_df.to_csv(filename, index=False)
    print(f"\n[+] Saved full DataFrame to: {filename}")

    log_dir = "output/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = f"{log_dir}/deepsight_{entity_type}_{kpi_column}_{timestamp}.log"

    anomalies_data = []

    with open(log_path, 'w') as log_file:
        def log_print(*args):
            line = " ".join(map(str, args)) + "\n"
            print(line.strip())
            log_file.write(line)

        log_print(f"[+] Running DeepSight analysis on {entity_type} data with {mindset} mindset...")
        log_print(f"[!] Rule-Based Anomalies: {len(rule_anomalies)}")
        log_print(f"[!] ML-Based Anomalies:   {len(ml_anomalies)}")

        combined_anomalies = pd.concat([rule_anomalies, ml_anomalies]).drop_duplicates()

        if not combined_anomalies.empty:
            for _, row in combined_anomalies.iterrows():
                log_print(f"\nAnomaly Date: {row['Month']}, Value: {row[kpi_column]}, Change: {row['pct_change']:.2%}")
                hypotheses = generate_hypotheses(row, kpi_column, SCHEMA_SUMMARY)
                log_print("\nRaw LLM Output:\n", hypotheses)

                sql_queries = parse_hypotheses_and_queries(hypotheses)
                log_print("SQL Queries:", sql_queries)

                if not sql_queries:
                    log_print("⚠️ No SQL queries found in LLM response.")
                    continue

                log_print(f"\n[+] Validating {len(sql_queries)} potential causes...\n")

                validation_results = []

                for item in sql_queries:
                    log_print(f"\nPotential Cause #{item['hypothesis_number']}: {item['hypothesis']}")
                    log_print("\nSQL Query:")
                    log_print(item['query'])

                    clean_query = sanitize_sql_query(item['query'])
                    result = execute_sql_query(clean_query)

                    if isinstance(result, dict) and 'error' in result and "must appear in the GROUP BY clause or be used in an aggregate function" in result['error']:
                        log_print("⚠️ Detected GROUP BY error, attempting to fix query...")
                        fixed_query = fix_group_by_error(clean_query, result['error'])
                        log_print("Fixed Query:")
                        log_print(fixed_query)
                        result = execute_sql_query(fixed_query)
                        log_print("Retry Result:", result)

                    if isinstance(result, dict) and 'error' in result:
                        log_print("❌ Error:", result['error'])
                        log_print("Analysis: Could not validate due to error.")
                        log_print("Recommendation: Review query for syntax or schema issues.\n")
                        validation_results.append({
                            "hypothesis_number": item['hypothesis_number'],
                            "hypothesis": item['hypothesis'],
                            "query": item['query'],
                            "result_count": 0,
                            "error": result,
                            "results": []
                        })
                    else:
                        result_count = len(result)
                        status = "Supported" if result_count > 0 else "Not Supported"
                        log_print("✅ Result Sample:", result[:5])
                        log_print(f"Analysis: {status}. {'Data confirms the cause.' if result_count > 0 else 'No data supports this cause.'}")
                        log_print(f"Recommendation: {'Investigate further or act on this cause.' if result_count > 0 else 'Explore alternative causes.'}\n")
                        validation_results.append({
                            "hypothesis_number": item['hypothesis_number'],
                            "hypothesis": item['hypothesis'],
                            "query": item['query'],
                            "result_count": result_count,
                            "error": None,
                            "results": result
                        })

                anomalies_data.append({
                    "row": row,
                    "validation_results": validation_results
                })

            if anomalies_data:
                log_print("\n[+] Generating combined anomalies report...")
                combined_insight = generate_combined_insight(anomalies_data, kpi_column, entity_type, mindset)
                log_print("\n[+] Combined Anomalies Report:")
                log_print(combined_insight)
                save_markdown_report(combined_insight, entity_type, kpi_column)
        else:
            log_print("[✓] No significant anomalies found.")