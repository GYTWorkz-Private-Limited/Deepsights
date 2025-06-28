import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration using environment variables
DATABASE_CONFIG = {
    "host": os.getenv("DATABASE_HOST"),
    "port": int(os.getenv("DATABASE_PORT")),
    "database": os.getenv("DATABASE_NAME"),
    "user": os.getenv("DATABASE_USER"),
    "password": os.getenv("DATABASE_PASSWORD")
}

# Schema summary for LLM context
SCHEMA_SUMMARY = """
Database Schema: vw_ai_rpt_pnl

Table: vw_ai_rpt_pnl
Columns:
- realm_id (VARCHAR): Unique identifier for the business realm
- Month (DATE): Month of the transaction
- Customer (VARCHAR): Customer identifier
- Vendor (VARCHAR): Vendor identifier
- Revenue (DECIMAL): Revenue amount
- Expense (DECIMAL): Expense amount
- Account Sub Type (VARCHAR): Sub-category of account (e.g., SalesOfProductIncome, ServiceFeeIncome)
- Account (VARCHAR): Main account category
- Transaction Type (VARCHAR): Type of transaction
- PNL Type (VARCHAR): Profit and Loss category

Key Relationships:
- Each row represents a monthly aggregated transaction
- Revenue and Expense are mutually exclusive (one will be 0)
- Account Sub Type is a subdivision of Account
- Customer and Vendor are mutually exclusive based on transaction type
"""

# Application configuration
APP_CONFIG = {
    "debug": os.getenv("DEBUG", "True").lower() == "true",
    "log_level": os.getenv("LOG_LEVEL", "INFO"),
    "max_workers": int(os.getenv("MAX_WORKERS", 6)),
    "timeout_seconds": int(os.getenv("TIMEOUT_SECONDS", 300))
}

# Output configuration
OUTPUT_CONFIG = {
    "base_dir": os.getenv("OUTPUT_DIR", "output"),
    "reports_dir": os.getenv("REPORTS_DIR", "output/reports"),
    "logs_dir": os.getenv("LOGS_DIR", "output/logs"),
    "anomalies_dir": os.getenv("ANOMALIES_DIR", "output/anomalies")
}

# ML configuration
ML_CONFIG = {
    "contamination": float(os.getenv("ML_CONTAMINATION", 0.01)),
    "random_state": int(os.getenv("ML_RANDOM_STATE", 42)),
    "rolling_window": int(os.getenv("ROLLING_WINDOW", 3))
}

# Analysis configuration
ANALYSIS_CONFIG = {
    "default_mindset": os.getenv("DEFAULT_MINDSET", "growth"),
    "default_entity_type": os.getenv("DEFAULT_ENTITY_TYPE", "Account Sub Type"),
    "default_kpi_column": os.getenv("DEFAULT_KPI_COLUMN", "Revenue"),
    "realm_id": os.getenv("REALM_ID", "999999999")
}

# OpenAI configuration
OPENAI_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY"),
    "model": os.getenv("OPENAI_MODEL", "gpt-4"),
    "temperature": float(os.getenv("OPENAI_TEMPERATURE", 0.3)),
    "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", 2000))
}
