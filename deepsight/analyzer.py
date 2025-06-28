import re
import os
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
from sqlparse import parse, tokens
from openai import OpenAI
from .anomaly_detector import detect_monthly_kpi_anomalies
from .hypothesis_generator import generate_hypotheses
from .query_executor import execute_sql_query
from .config import SCHEMA_SUMMARY, OUTPUT_CONFIG, ANALYSIS_CONFIG

# Load environment variables
load_dotenv()

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

# SCHEMA_SUMMARY is now imported from config.py

def save_anomalies_to_csv(anomalies_df, entity_type, kpi_column, output_dir="anomalies_output"):
    """
    Save anomalies DataFrame to a CSV file with timestamped filename.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/anomalies_{entity_type}_{kpi_column}_{timestamp}.csv"
    anomalies_df.to_csv(filename, index=False)
    print(f"\n[+] Saved anomalies to: {filename}")

def save_markdown_report(content, entity_type, kpi_column, output_dir=None):
    """
    Save markdown report with timestamped filename.
    """
    if output_dir is None:
        output_dir = OUTPUT_CONFIG['reports_dir']

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/combined_anomalies_report_{entity_type}_{kpi_column}_{timestamp}.md"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"\n[+] Saved Combined Anomalies Report to: {filename}")
    return filename

def generate_combined_insight(anomalies_data, kpi_column, entity_type, mindset="growth"):
    """
    Generate a comprehensive insight report with actual data and numbers.
    """
    if not anomalies_data:
        return "## No Anomalies Detected\n\nNo significant anomalies were found in the analysis period."

    # Calculate accurate statistics
    total_value = sum(anomaly['row'][kpi_column] for anomaly in anomalies_data)

    # Get actual entity data with proper calculations
    entity_details = {}
    time_periods = set()

    for anomaly in anomalies_data:
        row = anomaly['row']
        entity_name = row.get(entity_type, "Unknown")
        month = str(row['Month'])[:7]  # YYYY-MM format
        time_periods.add(month)

        if entity_name not in entity_details:
            entity_details[entity_name] = {
                'total_value': 0,
                'anomalies': [],
                'months': set()
            }

        entity_details[entity_name]['total_value'] += row[kpi_column]
        entity_details[entity_name]['months'].add(month)
        entity_details[entity_name]['anomalies'].append({
            'month': month,
            'value': row[kpi_column],
            'change': row['pct_change'] if pd.notna(row['pct_change']) else 0,
            'validation_count': len(anomaly.get('validation_results', []))
        })

    # Sort entities by total value
    top_entities = sorted(entity_details.items(), key=lambda x: x[1]['total_value'], reverse=True)

    # Calculate overall statistics
    total_anomalies = len(anomalies_data)
    time_span = f"{min(time_periods)} to {max(time_periods)}" if len(time_periods) > 1 else list(time_periods)[0]

    # Calculate average change (weighted by value)
    weighted_changes = []
    for anomaly in anomalies_data:
        row = anomaly['row']
        if pd.notna(row['pct_change']):
            weighted_changes.append(row['pct_change'] * row[kpi_column])

    avg_change = (sum(weighted_changes) / total_value) if total_value > 0 and weighted_changes else 0

    # Create detailed breakdown for top entities
    entity_breakdown = []
    for i, (entity_name, data) in enumerate(top_entities[:5]):
        avg_entity_change = sum(a['change'] for a in data['anomalies']) / len(data['anomalies'])
        entity_breakdown.append({
            'rank': i + 1,
            'name': entity_name,
            'value': data['total_value'],
            'percentage_of_total': (data['total_value'] / total_value) * 100,
            'avg_change': avg_entity_change,
            'anomaly_count': len(data['anomalies']),
            'months_affected': len(data['months'])
        })

    # Generate the report using actual data
    report = f"""## Executive Summary

During the analysis period ({time_span}), we identified **{total_anomalies} significant anomalies** across {entity_type} data, representing a total {kpi_column} of **${total_value:,.0f}** with an average weighted change of **{avg_change:.1%}**.

## Key Anomaly Contributors

"""

    for entity in entity_breakdown[:3]:
        report += f"""### {entity['rank']}. {entity['name']}
- **Total {kpi_column}**: ${entity['value']:,.0f} ({entity['percentage_of_total']:.1f}% of total anomalies)
- **Average Change**: {entity['avg_change']:.1%}
- **Anomalies**: {entity['anomaly_count']} occurrences across {entity['months_affected']} months

"""

    # Add validation insights if available
    validated_insights = []
    for anomaly in anomalies_data[:3]:
        if anomaly.get('validation_results'):
            for validation in anomaly['validation_results'][:2]:
                if validation.get('result_count', 0) > 0:
                    validated_insights.append(validation.get('hypothesis', 'Validation confirmed'))

    if validated_insights:
        report += "## Validated Insights\n\n"
        for i, insight in enumerate(validated_insights[:3], 1):
            report += f"{i}. {insight}\n"
        report += "\n"

    # Add business recommendations
    if avg_change > 0:
        trend_desc = "positive growth trends"
        recommendation = f"capitalize on the growth momentum in top-performing {entity_type.lower()}s"
    else:
        trend_desc = "concerning decline patterns"
        recommendation = f"investigate and address the underlying issues affecting {entity_type.lower()} performance"

    report += f"""## Business Insights & Recommendations

The analysis reveals {trend_desc} with {len(top_entities)} {entity_type.lower()}s showing significant variations. The top 3 contributors account for {sum(e['percentage_of_total'] for e in entity_breakdown[:3]):.1f}% of total anomalous {kpi_column}.

**Key Recommendation**: Focus immediate attention on {top_entities[0][0]} and {top_entities[1][0] if len(top_entities) > 1 else 'other top contributors'} to {recommendation}. Implement enhanced monitoring for these entities to detect future anomalies early.

**Next Steps**:
1. Deep-dive analysis on the top 3 contributors
2. Establish automated alerts for similar patterns
3. Review operational processes affecting these entities
"""

    return report

def run_deepsight_analysis(df, kpi_column, entity_type, mindset=None):
    """
    Run DeepSight analysis with a combined anomalies report in the desired format.
    """
    if mindset is None:
        mindset = ANALYSIS_CONFIG['default_mindset']

    print(f"\n[+] Running DeepSight analysis on {entity_type} data with {mindset} mindset...")
    result = detect_monthly_kpi_anomalies(df, kpi_column)
    full_df = result['full_df']
    rule_anomalies = result['rule_based_anomalies']
    ml_anomalies = result['ml_based_anomalies']

    # Use config for output directories
    output_dir = OUTPUT_CONFIG['anomalies_dir']
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/flagged_data_{entity_type}_{kpi_column}_{timestamp}.csv"
    full_df.to_csv(filename, index=False)
    print(f"\n[+] Saved full DataFrame to: {filename}")

    log_dir = OUTPUT_CONFIG['logs_dir']
    os.makedirs(log_dir, exist_ok=True)
    log_path = f"{log_dir}/deepsight_{entity_type}_{kpi_column}_{timestamp}.log"

    anomalies_data = []

    with open(log_path, 'w', encoding='utf-8') as log_file:
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