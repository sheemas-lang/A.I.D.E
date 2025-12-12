"""
Project configuration for paths, features, and risk settings.
"""
import os

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "credit_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "loan_model.pkl")

# Feature order used for both training and inference
FEATURE_ORDER = [
    "age",
    "monthly_income",
    "loan_amount",
    "employment_years",
    "debt_to_income",
    "loan_to_income",
    "employment_stability",
    "existing_loans",
]

# Loan / risk settings
BASE_INTEREST_RATE = 12.0  # base APR
RISK_PREMIUM = {
    "low": 0.0,
    "medium": 2.0,
    "high": 5.0,
    "very_high": 8.0,
}

