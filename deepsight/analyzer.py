

# deepsight.py
import re
from datetime import datetime
from .anomaly_detector import detect_monthly_kpi_anomalies
from .hypothesis_generator import generate_hypotheses
from .query_executor import execute_sql_query
import os
from datetime import datetime
import pandas as pd
import sys

    
    
def parse_hypotheses_and_queries(raw_response):
    import re

    # Match Explanation N: followed by explanation and then Query:
    pattern = r"(?:Explanation|Hypothesis)\s+(\d+):\s*(.*?)\s*Query:\s*((?:.|\n)*?)(?=(?:\n(?:Explanation|Hypothesis)\s+\d+:|\Z))"
    matches = re.findall(pattern, raw_response, re.DOTALL)

    results = []
    for hyp_num, hypothesis, query in matches:
        # Clean up query (remove markdown if any)
        query = re.sub(r'```sql\n|```', '', query).strip()
        results.append({
            "hypothesis_number": int(hyp_num),
            "hypothesis": hypothesis.strip(),
            "query": query.strip()
        })

    return results

   

def sanitize_sql_query(query):
    # query = re.sub(r'"([^"]+)"', r'\1', query)
    query = query.replace("your_table_name", "vw_ai_rpt_pnl") \
        .replace("financial_data", "vw_ai_rpt_pnl") \
        .replace("customer", "Customer") \
        .replace("vendor", "Vendor") \
        .replace("Month", "Month") \
        .replace("Revenue", "Revenue") \
        .replace("Expense", "Expense") 
    return query.strip()

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
    
from openai import OpenAI



def save_markdown_report(content, entity_type, kpi_column, output_dir="output/reports"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/deepsight_{entity_type}_{kpi_column}_{timestamp}.md"

    with open(filename, 'w') as f:
        f.write(content)

    print(f"\n[+] Saved Markdown report to: {filename}")
    return filename


def generate_final_insight(anomaly_row, kpi_column, hypotheses_data):
    """
    Sends anomaly + hypotheses + SQL results back to LLM for final insight
    """
    prompt = f"""
You are a financial analyst who has analyzed the following anomaly:

KPI: {kpi_column}
Date: {anomaly_row['Month']}
Value: {anomaly_row[kpi_column]}
Change: {anomaly_row['pct_change']:.2%} compared to rolling average

Hypotheses and Validation Results:
"""

    for item in hypotheses_data:
        status = "Supported" if item['result_count'] > 0 else "Not Supported"
        if isinstance(item['error'], dict) and 'error' in item['error']:
            status = "Error"
        
        prompt += f"""
Hypothesis #{item['hypothesis_number']}:
{item['hypothesis']}

Query:
{item['query']}

Validation Result:
{status}
"""

    prompt += """
Based on the above, provide a final insight summarizing which hypotheses are supported, 
which are not, and what likely caused the anomaly. Use business terms and avoid technical jargon.
"""

    client = OpenAI(api_key="Api_key")
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content



def run_deepsight_analysis(df, kpi_column, entity_type):
    print(f"\n[+] Running DeepSight analysis on {entity_type} data...")

    result = detect_monthly_kpi_anomalies(df, kpi_column)
    full_df = result['full_df']
    rule_anomalies = result['rule_based_anomalies']
    ml_anomalies = result['ml_based_anomalies']

    # Save full DataFrame
    output_dir = "output/anomalies"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/flagged_data_{entity_type}_{kpi_column}_{timestamp}.csv"
    full_df.to_csv(filename, index=False)
    print(f"\n[+] Saved full DataFrame to: {filename}")

    # Save log output to file
    log_dir = "output/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = f"{log_dir}/deepsight_{entity_type}_{kpi_column}_{timestamp}.log"

    with open(log_path, 'w') as log_file:
        def log_print(*args):
            line = " ".join(map(str, args)) + "\n"
            print(line.strip())
            log_file.write(line)

        log_print(f"[+] Running DeepSight analysis on {entity_type} data...")
        log_print(f"[!] Rule-Based Anomalies: {len(rule_anomalies)}")
        log_print(f"[!] ML-Based Anomalies:   {len(ml_anomalies)}")

        combined_anomalies = pd.concat([rule_anomalies, ml_anomalies]).drop_duplicates()

        if not combined_anomalies.empty:
            for _, row in combined_anomalies.iterrows():
                log_print(f"\nAnomaly Date: {row['Month']}, Value: {row[kpi_column]}, Change: {row['pct_change']:.2%}")

                hypotheses = generate_hypotheses(row, kpi_column, SCHEMA_SUMMARY)
                log_print("\nRaw LLM Output:\n", hypotheses)

                sql_queries = parse_hypotheses_and_queries(hypotheses)
                log_print("SQL Query", sql_queries)

                if not sql_queries:
                    log_print("⚠️ No SQL queries found in LLM response.")
                    continue

                log_print(f"\n[+] Validating {len(sql_queries)} hypotheses...\n")

                # Store validation results
                validation_results = []

                for item in sql_queries:
                    log_print(f"\nHypothesis #{item['hypothesis_number']}")
                    log_print(item['hypothesis'])
                    log_print("\nSQL Query:")
                    log_print(item['query'])

                    clean_query = sanitize_sql_query(item['query'])
                    log_print("clean", clean_query)

                    result = execute_sql_query(clean_query)

                    if isinstance(result, dict) and 'error' in result:
                        log_print("❌ SQL Error:", result['error'])
                        log_print("Status: ❌ Could not validate hypothesis due to error.\n")
                        validation_results.append({
                            "hypothesis_number": item['hypothesis_number'],
                            "hypothesis": item['hypothesis'],
                            "query": item['query'],
                            "result_count": 0,
                            "error": result
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
                            "error": result if isinstance(result, dict) else None
                        })

                # 🧠 Send to LLM for final insight
                final_insight = generate_final_insight(row, kpi_column, validation_results)
                log_print("\n[+] Final Insight from LLM:")
                log_print(final_insight)
                log_print("-" * 60)
        else:
            log_print("[✓] No significant anomalies found.")