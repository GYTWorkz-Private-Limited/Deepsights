# main.py
from deepsight import run_deepsight_analysis
from deepsight.data_filter import fetch_data_for_realm

realm_id = "999999999"
revenue_df, expense_df, AS_df_revenue, AS_df_expense, A_df_revenue, A_df_expense = fetch_data_for_realm(realm_id)

# print(AS_df_revenue)
# print(A_df_revenue)

# print(AS_df_expense)
# print(A_df_expense)
# if not revenue_df.empty:
#     # revenue_df.to_csv('output/revenue_df.csv', index=False)
#     run_deepsight_analysis(revenue_df, kpi_column='Revenue', entity_type='Customer')

# if not expense_df.empty:
#    # expense_df.to_csv('output/expense_df.csv', index=False)
#     run_deepsight_analysis(expense_df, kpi_column='Expense', entity_type='Vendor')

if not AS_df_revenue.empty:
    # AS_df_revenue.to_csv('output/AS_df_revenue.csv', index=False)
    run_deepsight_analysis(AS_df_revenue, kpi_column='Revenue', entity_type='Account Sub Type')
    
# if not AS_df_expense.empty:
#     # AS_df_expense.to_csv('output/AS_df_expense.csv', index=False)
#     run_deepsight_analysis(AS_df_expense, kpi_column='Expense', entity_type='Account Sub Type')
    
# if not A_df_revenue.empty:
#     # A_df_revenue.to_csv('output/A_df_revenue.csv', index=False)
#     run_deepsight_analysis(A_df_revenue, kpi_column='Revenue', entity_type='Account')
    
# if not A_df_expense.empty:
#     # A_df_expense.to_csv('output/A_df_expense.csv', index=False)
#     run_deepsight_analysis(A_df_expense, kpi_column='Expense', entity_type='Account')
