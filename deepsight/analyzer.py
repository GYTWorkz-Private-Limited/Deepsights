

#+++++++++####################+++++++++++++++++++++++++++
import re
from datetime import datetime
from .anomaly_detector import detect_monthly_kpi_anomalies
from .hypothesis_generator import generate_hypotheses
from .query_executor import execute_sql_query
import os
import pandas as pd
import sys
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
        # Extract the problematic column from the error message
        match = re.search(r'column "([^"]+)" must appear', error_message)
        if not match:
            return query  # Return original query if column not found
        
        problematic_column = match.group(1)  # e.g., "vw_ai_rpt_pnl.Month"
        column_name = problematic_column.split('.')[-1]  # Extract just "Month"
        
        # Find the HAVING clause and move the condition to WHERE
        having_pattern = r'(HAVING\s+)(.*?)(;|\Z)'
        having_match = re.search(having_pattern, query, re.IGNORECASE | re.DOTALL)
        if not having_match:
            return query  # Return original query if no HAVING clause
        
        having_clause = having_match.group(2)
        # Extract the condition involving the problematic column (e.g., "Month" = '2024-10-01')
        condition_pattern = rf'\b{column_name}\b\s*(=|>|<|>=|<=|!=)\s*\'[^\']+\''
        condition_match = re.search(condition_pattern, having_clause, re.IGNORECASE)
        if not condition_match:
            return query
        
        condition = condition_match.group(0)
        
        # Remove the condition from HAVING
        new_having = re.sub(rf'\b{condition}\b(\s*AND\s*)?', '', having_clause, flags=re.IGNORECASE).strip()
        if new_having and not new_having.lower().startswith('and'):
            new_having = f"AND {new_having}"
        new_having = new_having.strip()
        
        # Add the condition to WHERE clause
        where_pattern = r'(WHERE\s+.*?)(GROUP\s+BY|HAVING|;|\Z)'
        where_match = re.search(where_pattern, query, re.IGNORECASE | re.DOTALL)
        if where_match:
            where_clause = where_match.group(1).strip()
            new_where = f"{where_clause} AND {condition}" if where_clause.lower() != 'where' else f"WHERE {condition}"
            new_query = re.sub(where_pattern, f"{new_where} \\2", query, flags=re.IGNORECASE | re.DOTALL)
        else:
            # If no WHERE clause, add one before GROUP BY
            new_query = re.sub(r'(FROM\s+.*?)(GROUP\s+BY)', f'\\1 WHERE {condition} \\2', query, flags=re.IGNORECASE | re.DOTALL)
        
        # Update HAVING clause
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
    filename = f"{output_dir}/deepsight_{entity_type}_{kpi_column}_{timestamp}.md"
    with open(filename, 'w') as f:
        f.write(content)
    print(f"\n[+] Saved Markdown report to: {filename}")
    return filename

def generate_final_insight(anomaly_row, kpi_column, hypotheses_data, mindset="growth"):
    """
    Generate a comprehensive insight report by sending anomaly, hypotheses, explanations, 
    SQL queries, and complete query results to the LLM, using the specified mindset.
    """
    mindset_guidance = {
        "growth": "Analyze the anomaly with a growth mindset, focusing on opportunities for expansion, new strategies, and positive outcomes. Highlight how the anomaly could lead to business growth or new market opportunities.",
        "risk": "Analyze the anomaly with a risk mindset, focusing on potential threats, vulnerabilities, and mitigation strategies. Highlight risks to financial stability or operations and suggest ways to address them.",
        "efficiency": "Analyze the anomaly with an efficiency mindset, focusing on process improvements, cost savings, and operational streamlining. Highlight ways to optimize resources or improve performance."
    }
    
    mindset_prompt = mindset_guidance.get(mindset.lower(), mindset_guidance["growth"])
    
    prompt = f"""
You are a financial analyst tasked with analyzing a financial anomaly using a {mindset} mindset. 
{mindset_prompt}

**Anomaly Details**:
- KPI: {kpi_column}
- Date: {anomaly_row['Month']}
- Value: {anomaly_row[kpi_column]}
- Change: {anomaly_row['pct_change']:.2%} compared to rolling average

**Hypotheses and Validation Results**:
"""
    for item in hypotheses_data:
        status = "Supported" if item['result_count'] > 0 else "Not Supported"
        if isinstance(item['error'], dict) and 'error' in item['error']:
            status = "Error"
            query_result = f"Error: {item['error']['error']}"
        else:
            query_result = f"{item['result_count']} rows returned: {item['results']}"
        
        prompt += f"""
**Hypothesis #{item['hypothesis_number']}**:
{item['hypothesis']}

**SQL Query**:
```sql
{item['query']}
```

**Validation Result**:
- Status: {status}
- Query Output: {query_result}
"""
    prompt += f"""
**Task**:
Analyze the anomaly by synthesizing *all* query results collectively to provide a cohesive explanation. Do not analyze each hypothesis individually; instead, connect patterns, correlations, or contradictions across all results to determine the likely cause(s) of the anomaly. Use business terms and avoid technical jargon. The report should be structured in markdown with the following sections:

1. **Summary**: Briefly describe the anomaly and its significance in the context of the business.
2. **Analysis**: Provide a comprehensive analysis by combining insights from all query results. Identify patterns or trends across the results (e.g., specific customers, transaction types, or accounts contributing to the anomaly). Highlight any correlations or contradictions that inform the anomaly's cause.
3. **Likely Cause**: Summarize the most likely cause(s) of the anomaly, supported by the collective evidence from the query results.
4. **Recommendations**: Provide actionable recommendations aligned with the {mindset} mindset to address the anomaly or leverage it for business advantage.
5. **Conclusion**: Summarize key findings and suggest next steps for further investigation or action.

Ensure the analysis is thorough, connects the dots across all query results, and aligns with the {mindset} mindset. Format the response in markdown with clear headings.

"""
    client = OpenAI(api_key="Api_key")
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}]
    )
    
    report_content = response.choices[0].message.content
    return report_content

def run_deepsight_analysis(df, kpi_column, entity_type, mindset="growth"):
    """
    Run DeepSight analysis on the provided DataFrame with specified mindset.
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
                log_print("SQL Query:", sql_queries)

                if not sql_queries:
                    log_print("⚠️ No SQL queries found in LLM response.")
                    continue

                log_print(f"\n[+] Validating {len(sql_queries)} hypotheses...\n")

                validation_results = []

                for item in sql_queries:
                    log_print(f"\nHypothesis #{item['hypothesis_number']}")
                    log_print(item['hypothesis'])
                    log_print("\nSQL Query:")
                    log_print(item['query'])

                    clean_query = sanitize_sql_query(item['query'])
                    result = execute_sql_query(clean_query)

                    # Check for GROUP BY-related error and attempt fix
                    if isinstance(result, dict) and 'error' in result and "must appear in the GROUP BY clause or be used in an aggregate function" in result['error']:
                        log_print("⚠️ Detected GROUP BY error, attempting to fix query...")
                        fixed_query = fix_group_by_error(clean_query, result['error'])
                        log_print("Fixed Query:")
                        log_print(fixed_query)

                        # Retry with fixed query
                        result = execute_sql_query(fixed_query)
                        log_print("Retry Result:", result)

                    if isinstance(result, dict) and 'error' in result:
                        log_print("❌ SQL Error:", result['error'])
                        log_print("Status: ❌ Could not validate hypothesis due to error.\n")
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
                        log_print("✅ Query Result Sample:", result[:5])
                        status = "✅ Supported" if result_count > 0 else "❌ Not Supported"
                        log_print(f"Status: {status}\n")

                        validation_results.append({
                            "hypothesis_number": item['hypothesis_number'],
                            "hypothesis": item['hypothesis'],
                            "query": item['query'],
                            "result_count": result_count,
                            "error": None,
                            "results": result
                        })

                final_insight = generate_final_insight(row, kpi_column, validation_results, mindset)
                log_print("\n[+] Final Insight Report from LLM:")
                log_print(final_insight)
                log_print("-" * 60)
                
                save_markdown_report(final_insight, entity_type, kpi_column)
        else:
            log_print("[✓] No significant anomalies found.")