"""
Credit Card Fraud Detection - Model Training Script
Trains RandomForestClassifier and GradientBoostingClassifier on financial transaction data
incorporating OrgDelta, DestDelta, balance math verification, and PaySim features.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLUMNS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "orgDelta",
    "destDelta",
    "absOrgDelta",
    "absDestDelta",
    "hasNegativeBalance",
    "insufficientFunds",
    "isSystemFlagged",
    "origDrained",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
]


def generate_synthetic_transactions(n_samples=30000, fraud_ratio=0.15, random_state=42):
    """
    Generates realistic credit card and bank transactions covering:
    - Valid balanced transactions (isFraud = 0)
    - Fraudulent transactions (isFraud = 1):
      1. Full balance drain attacks
      2. Sender balance math discrepancies (OrgDelta != 0)
      3. Recipient balance math discrepancies (DestDelta != 0)
      4. Insufficient sender balance (amount > oldbalanceOrg)
      5. Negative balance tampering (e.g. -0.07, -0.02)
      6. System threshold breaches (amount > $200,000)
    """
    np.random.seed(random_state)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    records = []
    all_types = ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]

    # 1. Generate Legitimate Transactions (Balanced math, positive balances, no drain)
    legit_probs = [0.20, 0.25, 0.35, 0.10, 0.10]
    for _ in range(n_legit):
        t_type = np.random.choice(all_types, p=legit_probs)
        step = int(np.random.randint(1, 744))
        amount = float(np.round(np.random.uniform(10.0, 150000.0), 2))

        if t_type == "CASH_IN":
            oldbalanceOrg = float(np.round(np.random.uniform(50.0, 200000.0), 2))
            newbalanceOrig = float(np.round(oldbalanceOrg + amount, 2))
            oldbalanceDest = float(np.round(amount + np.random.uniform(100.0, 300000.0), 2))
            newbalanceDest = float(np.round(oldbalanceDest - amount, 2))
        else:
            # Outgoing: sender has strictly more than amount so account is not drained to 0
            extra_buffer = float(np.round(np.random.uniform(25.0, 100000.0), 2))
            oldbalanceOrg = float(np.round(amount + extra_buffer, 2))
            newbalanceOrig = float(np.round(oldbalanceOrg - amount, 2))
            oldbalanceDest = float(np.round(np.random.uniform(0.0, 200000.0), 2))
            newbalanceDest = float(np.round(oldbalanceDest + amount, 2))

        records.append({
            "step": step,
            "type": t_type,
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,
            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest,
            "isFraud": 0,
        })

    # 2. Generate Fraudulent Transactions across diverse attack vectors
    attack_modes = [
        "account_drain",
        "sender_discrepancy",
        "recipient_discrepancy",
        "negative_balance",
        "insufficient_funds",
        "high_value_breach",
    ]

    for _ in range(n_fraud):
        t_type = np.random.choice(all_types, p=[0.35, 0.35, 0.15, 0.05, 0.10])
        step = int(np.random.randint(1, 744))
        mode = np.random.choice(attack_modes)

        if mode == "account_drain":
            oldbalanceOrg = float(np.round(np.random.uniform(500.0, 600000.0), 2))
            amount = oldbalanceOrg
            newbalanceOrig = 0.0
            oldbalanceDest = float(np.round(np.random.choice([0.0, np.random.uniform(100.0, 50000.0)]), 2))
            newbalanceDest = float(np.round(np.random.choice([0.0, oldbalanceDest + amount]), 2))

        elif mode == "sender_discrepancy":
            amount = float(np.round(np.random.uniform(100.0, 150000.0), 2))
            oldbalanceOrg = float(np.round(amount + np.random.uniform(100.0, 50000.0), 2))
            # Introduce wrong after-balance
            newbalanceOrig = float(np.round(oldbalanceOrg - amount * np.random.uniform(0.0, 0.7), 2))
            oldbalanceDest = float(np.round(np.random.uniform(0.0, 50000.0), 2))
            newbalanceDest = float(np.round(oldbalanceDest + amount, 2))

        elif mode == "recipient_discrepancy":
            amount = float(np.round(np.random.uniform(100.0, 150000.0), 2))
            oldbalanceOrg = float(np.round(amount + np.random.uniform(100.0, 50000.0), 2))
            newbalanceOrig = float(np.round(oldbalanceOrg - amount, 2))
            oldbalanceDest = float(np.round(np.random.uniform(0.0, 50000.0), 2))
            # Recipient balance does not reflect the amount transferred
            newbalanceDest = float(np.round(np.random.choice([0.0, oldbalanceDest, oldbalanceDest + amount * 2.5]), 2))

        elif mode == "negative_balance":
            amount = float(np.round(np.random.uniform(10.0, 50000.0), 2))
            oldbalanceOrg = float(np.round(np.random.uniform(-500.0, -0.01), 2))
            newbalanceOrig = float(np.round(np.random.choice([0.0, oldbalanceOrg - amount]), 2))
            oldbalanceDest = float(np.round(np.random.uniform(-100.0, 500.0), 2))
            newbalanceDest = float(np.round(np.random.uniform(100.0, 50000.0), 2))

        elif mode == "insufficient_funds":
            oldbalanceOrg = float(np.round(np.random.uniform(0.0, 5000.0), 2))
            amount = float(np.round(oldbalanceOrg + np.random.uniform(500.0, 100000.0), 2))
            newbalanceOrig = 0.0
            oldbalanceDest = float(np.round(np.random.uniform(0.0, 20000.0), 2))
            newbalanceDest = float(np.round(oldbalanceDest + amount, 2))

        else:  # high_value_breach (> $200,000)
            amount = float(np.round(np.random.uniform(200001.0, 900000.0), 2))
            oldbalanceOrg = float(np.round(amount + np.random.uniform(0.0, 50000.0), 2))
            newbalanceOrig = float(np.round(oldbalanceOrg - amount, 2))
            oldbalanceDest = float(np.round(np.random.uniform(0.0, 100000.0), 2))
            newbalanceDest = float(np.round(oldbalanceDest + amount, 2))

        records.append({
            "step": step,
            "type": t_type,
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,
            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest,
            "isFraud": 1,
        })

    df = pd.DataFrame(records)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df


def engineer_features(df):
    data = df.copy()

    # For CASH_IN, sender balance increases; for all other types, sender balance decreases
    is_cash_in = data["type"] == "CASH_IN"
    data["orgDelta"] = np.where(
        is_cash_in,
        np.round(data["newbalanceOrig"] - data["oldbalanceOrg"] - data["amount"], 2),
        np.round(data["oldbalanceOrg"] - data["newbalanceOrig"] - data["amount"], 2),
    )
    data["destDelta"] = np.where(
        is_cash_in,
        np.round(data["oldbalanceDest"] - data["newbalanceDest"] - data["amount"], 2),
        np.round(data["newbalanceDest"] - data["oldbalanceDest"] - data["amount"], 2),
    )

    data["absOrgDelta"] = np.abs(data["orgDelta"])
    data["absDestDelta"] = np.abs(data["destDelta"])

    data["hasNegativeBalance"] = np.where(
        (data["amount"] <= 0)
        | (data["oldbalanceOrg"] < 0)
        | (data["newbalanceOrig"] < 0)
        | (data["oldbalanceDest"] < 0)
        | (data["newbalanceDest"] < 0),
        1,
        0,
    )

    data["insufficientFunds"] = np.where(
        (~is_cash_in) & (data["oldbalanceOrg"] < data["amount"]), 1, 0
    )

    data["isSystemFlagged"] = np.where(data["amount"] > 200000, 1, 0)
    data["origDrained"] = np.where(
        (~is_cash_in) & (data["oldbalanceOrg"] > 0) & (data["newbalanceOrig"] == 0), 1, 0
    )

    for t in ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]:
        data[f"type_{t}"] = (data["type"] == t).astype(int)

    return data[FEATURE_COLUMNS]


def train_and_export():
    print("Generating comprehensive fraud & legitimate transaction dataset...")
    raw_df = generate_synthetic_transactions(n_samples=30000, fraud_ratio=0.15)

    X = engineer_features(raw_df)
    y = raw_df["isFraud"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print("Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=120,
        max_depth=14,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)

    print("Training Gradient Boosting Classifier...")
    gb_model = GradientBoostingClassifier(
        n_estimators=120,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
    )
    gb_model.fit(X_train, y_train)

    rf_preds = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    gb_preds = gb_model.predict(X_test)
    gb_probs = gb_model.predict_proba(X_test)[:, 1]

    print(f"RF Accuracy: {accuracy_score(y_test, rf_preds)*100:.2f}%, AUC: {roc_auc_score(y_test, rf_probs):.4f}")
    print(f"GB Accuracy: {accuracy_score(y_test, gb_preds)*100:.2f}%, AUC: {roc_auc_score(y_test, gb_probs):.4f}")

    joblib.dump(rf_model, os.path.join(MODEL_DIR, "rf_model.joblib"))
    joblib.dump(gb_model, os.path.join(MODEL_DIR, "gb_model.joblib"))
    print("Models saved successfully!")


if __name__ == "__main__":
    train_and_export()
