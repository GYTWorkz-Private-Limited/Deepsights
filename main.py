# # Example usage
# import deepsight

# deepsight.run_deepsight_analysis(999999999)


# main.py
from deepsight import run_deepsight_analysis
from deepsight.data_filter import fetch_data_for_realm

realm_id = "999999999"
revenue_df, expense_df = fetch_data_for_realm(realm_id)

if not revenue_df.empty:
    # revenue_df.to_csv('output/revenue_df.csv', index=False)
    run_deepsight_analysis(revenue_df, kpi_column='Revenue', entity_type='Customer')

# if not expense_df.empty:
   # # expense_df.to_csv('output/expense_df.csv', index=False)
#     run_deepsight_analysis(expense_df, kpi_column='Expense', entity_type='Vendor')