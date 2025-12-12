"""
Credit scoring utilities: load model, build features, predict, and basic risk logic.
"""
import os
import pickle
import numpy as np
import pandas as pd

from config import FEATURE_ORDER, MODEL_PATH, BASE_INTEREST_RATE, RISK_PREMIUM

# Cache for loaded artifacts
_MODEL = None


def load_model():
    """Load the trained model from disk (cached after first load)."""
    global _MODEL
    if _MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Train and save the model first."
            )
        with open(MODEL_PATH, "rb") as f:
            obj = pickle.load(f)
        # Support both raw model or dict {"model": model, "features": ...}
        _MODEL = obj.get("model") if isinstance(obj, dict) else obj
    return _MODEL


def _compute_features(app):
    """Compute derived features from raw application data."""
    income = float(app.get("monthly_income", 0))
    loan_amount = float(app.get("loan_amount", 0))
    monthly_expenses = float(app.get("monthly_expenses", income * 0.5))
    age = float(app.get("age", 0))
    employment_years = float(app.get("employment_years", 0))
    existing_loans = float(app.get("existing_loans", 0))

    debt_to_income = (monthly_expenses / income * 100) if income > 0 else 100.0
    loan_to_income = (loan_amount / (income * 12) * 100) if income > 0 else 999.0
    employment_stability = (employment_years / age) if age > 0 else 0.0

    # Build feature dict in expected order
    features = {
        "age": age,
        "monthly_income": income,
        "loan_amount": loan_amount,
        "employment_years": employment_years,
        "debt_to_income": debt_to_income,
        "loan_to_income": loan_to_income,
        "employment_stability": employment_stability,
        "existing_loans": existing_loans,
    }
    return features


def predict_loan_approval(app_data):
    """
    Predict approval using the trained model.
    Returns dict: approved (bool), approval_probability, features_used.
    """
    try:
        model = load_model()
        feat = _compute_features(app_data)
        row = pd.DataFrame([[feat[c] for c in FEATURE_ORDER]], columns=FEATURE_ORDER)
        proba = model.predict_proba(row)[0]
        pred = model.predict(row)[0]
        prob_approve = float(proba[1]) if len(proba) > 1 else float(proba[0])
        return {
            "approved": bool(pred),
            "approval_probability": prob_approve,
            "rejection_probability": 1.0 - prob_approve,
            "features_used": feat,
            "method": "model",
        }
    except Exception:
        # Fallback: simple rule-based decision
        return predict_rule_based(app_data)


def predict_rule_based(app):
    """Fallback rule-based approval."""
    feat = _compute_features(app)
    score = 0.0
    income = feat["monthly_income"]
    dti = feat["debt_to_income"]
    age = feat["age"]
    emp = feat["employment_years"]
    lti = feat["loan_to_income"]
    existing = feat["existing_loans"]

    if income >= 50000:
        score += 0.3
    elif income >= 30000:
        score += 0.2
    elif income >= 20000:
        score += 0.1

    if 25 <= age <= 60:
        score += 0.2

    if dti < 40:
        score += 0.2
    elif dti < 60:
        score += 0.1

    if emp >= 2:
        score += 0.2

    if lti < 200:
        score += 0.1

    if existing > 2:
        score -= 0.1

    approved = score >= 0.5
    prob = min(max(score, 0.0), 1.0)
    return {
        "approved": approved,
        "approval_probability": prob,
        "rejection_probability": 1.0 - prob,
        "features_used": feat,
        "method": "rule_based",
    }


def calculate_risk_level(app_data, prediction):
    """Assign a simple risk bucket based on probability and DTI."""
    prob = prediction.get("approval_probability", 0.5)
    dti = prediction.get("features_used", {}).get(
        "debt_to_income", app_data.get("debt_to_income", 50)
    )
    if prob >= 0.8 and dti < 40:
        return "low"
    if prob >= 0.6 and dti < 60:
        return "medium"
    if prob >= 0.4:
        return "high"
    return "very_high"


def calculate_interest_rate(risk_level):
    """Compute interest rate from base and risk premium."""
    return BASE_INTEREST_RATE + RISK_PREMIUM.get(risk_level, 5.0)


def calculate_emi(loan_amount, annual_rate, tenure_years):
    """Calculate EMI using the standard amortization formula."""
    if loan_amount <= 0 or annual_rate <= 0 or tenure_years <= 0:
        return 0.0
    r = annual_rate / (12 * 100)
    n = tenure_years * 12
    if r == 0:
        return loan_amount / n
    emi = (loan_amount * r * (1 + r) ** n) / ((1 + r) ** n - 1)
    return round(emi, 2)


def get_shap_explanation(app_data):
    """
    Optional SHAP explanation. Requires shap installed.
    Returns a dict with feature_importance if successful; otherwise a fallback message.
    """
    try:
        import shap

        model = load_model()
        feat = _compute_features(app_data)
        row = pd.DataFrame([[feat[c] for c in FEATURE_ORDER]], columns=FEATURE_ORDER)
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(row)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]  # positive class
        importance = dict(zip(FEATURE_ORDER, shap_vals[0]))
        return {"feature_importance": importance, "method": "shap"}
    except Exception as e:
        return {"message": f"SHAP not available: {e}", "method": "fallback"}

