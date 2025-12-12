# train_scratch.py (structure)

import os, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from xgboost import XGBClassifier  

# ---------- Config ----------
RANDOM_SEED = 42
DATA_PATH = "credit_data.csv"
MODEL_PATH = "models/loan_model.pkl"
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

# ---------- 1) Data loading / synthetic generation ----------
def load_or_make_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run make_dataset.py to generate it."
        )
    return df

# ---------- 2) Feature engineering ----------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    inc = df["monthly_income"].clip(lower=1)
    df["debt_to_income"] = (df["monthly_expenses"] / inc * 100).clip(0, 300)
    df["loan_to_income"] = (df["loan_amount"] / (inc * 12) * 100).clip(0, 500)
    df["employment_stability"] = (df["employment_years"] / df["age"]).clip(0, 1)
    return df

def select_features(df: pd.DataFrame):
    X = df[FEATURE_ORDER]
    y = df["approved"]
    return X, y

# ---------- 3) Train/test split ----------
def split_data(X, y):
    return train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

# ---------- 4) Model training ----------
def train_model(X_train, y_train):
    model = XGBClassifier(
        max_depth=4,
        n_estimators=200,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)
    return model

# ---------- 5) Evaluation ----------
def evaluate(model, X_train, y_train, X_test, y_test):
    for split_name, X, y in [
        ("train", X_train, y_train),
        ("test", X_test, y_test),
    ]:
        preds = model.predict(X)
        proba = model.predict_proba(X)[:, 1]
        acc = accuracy_score(y, preds)
        auc = roc_auc_score(y, proba)
        print(f"{split_name} accuracy: {acc:.3f} | AUC: {auc:.3f}")
    print("\nClassification report (test):")
    print(classification_report(y_test, model.predict(X_test)))

# ---------- 6) Save artifacts ----------
def save_model(model):
    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "features": FEATURE_ORDER}, f)
    print(f"Saved model to {MODEL_PATH}")

# ---------- 7) Main flow ----------
def main():
    np.random.seed(RANDOM_SEED)
    df = load_or_make_data()
    df = add_features(df)
    X, y = select_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    evaluate(model, X_train, y_train, X_test, y_test)
    save_model(model)

if __name__ == "__main__":
    main()