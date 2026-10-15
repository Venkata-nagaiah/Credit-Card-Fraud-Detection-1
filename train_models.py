"""
Credit Card Fraud Detection - Model Training Script
Trains RandomForestClassifier and GradientBoostingClassifier on financial transaction data.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
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


def generate_synthetic_transactions(n_samples=32000, fraud_ratio=0.20, random_state=42):
    np.random.seed(random_state)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    records = []
    all_types = ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]

    # 1. Generate Legitimate Transactions (isFraud = 0)
    for _ in range(n_legit):
        t_type = np.random.choice(all_types, p=[0.25, 0.25, 0.30, 0.10, 0.10])
        step = int(np.random.randint(1, 744))
        amount = float(np.round(np.random.uniform(5.0, 80000.0), 2))

        if t_type == "CASH_IN":
            oldbalanceOrg = float(np.round(np.random.uniform(50.0, 100000.0), 2))
            newbalanceOrig = float(np.round(oldbalanceOrg + amount, 2))
            oldbalanceDest = float(np.round(np.random.choice([0.0, amount + np.random.uniform(10.0, 50000.0)]), 2))
            newbalanceDest = float(np.round(max(0.0, oldbalanceDest - amount), 2)) if oldbalanceDest > 0 else 0.0
        else:
            # Outgoing payment or transfer: valid sender balance
            extra = float(np.round(np.random.uniform(10.0, 50000.0), 2))
            oldbalanceOrg = float(np.round(amount + extra, 2))
            newbalanceOrig = float(np.round(oldbalanceOrg - amount, 2))

            # In card payments or ATM, destination is often untracked (0.0 -> 0.0)
            if t_type in ["PAYMENT", "DEBIT"] or np.random.rand() > 0.5:
                oldbalanceDest = 0.0
                newbalanceDest = 0.0
            else:
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
            "isFraud": 0,
        })

    # 2. Generate Fraudulent Transactions (isFraud = 1)
    attack_modes = [
        "sender_mismatch",
        "insufficient_balance",
        "negative_balance",
        "massive_drain",
        "system_limit",
        "tracked_dest_mismatch",
    ]

    for _ in range(n_fraud):
        t_type = np.random.choice(all_types, p=[0.35, 0.30, 0.15, 0.10, 0.10])
        step = int(np.random.randint(1, 744))
        mode = np.random.choice(attack_modes)

        if mode == "sender_mismatch":
            amount = float(np.round(np.random.uniform(50.0, 50000.0), 2))
            oldbalanceOrg = float(np.round(amount + np.random.uniform(100.0, 20000.0), 2))
            # New balance is wrong (e.g. didn't decrease by amount)
            newbalanceOrig = float(np.round(oldbalanceOrg - amount * np.random.uniform(0.0, 0.6), 2))
            oldbalanceDest = 0.0
            newbalanceDest = 0.0

        elif mode == "insufficient_balance":
            oldbalanceOrg = float(np.round(np.random.uniform(10.0, 1000.0), 2))
            amount = float(np.round(oldbalanceOrg + np.random.uniform(200.0, 50000.0), 2))
            newbalanceOrig = 0.0
            oldbalanceDest = 0.0
            newbalanceDest = 0.0

        elif mode == "negative_balance":
            amount = float(np.round(np.random.uniform(10.0, 10000.0), 2))
            oldbalanceOrg = float(np.round(np.random.uniform(-500.0, -0.01), 2))
            newbalanceOrig = 0.0
            oldbalanceDest = float(np.round(np.random.uniform(-100.0, 0.0), 2))
            newbalanceDest = float(np.round(np.random.uniform(100.0, 10000.0), 2))

        elif mode == "massive_drain":
            amount = float(np.round(np.random.uniform(20000.0, 200000.0), 2))
            oldbalanceOrg = amount
            newbalanceOrig = 0.0
            oldbalanceDest = 0.0
            newbalanceDest = 0.0

        elif mode == "system_limit":
            amount = float(np.round(np.random.uniform(200001.0, 999999.0), 2))
            oldbalanceOrg = float(np.round(amount + 5000.0, 2))
            newbalanceOrig = float(np.round(oldbalanceOrg - amount, 2))
            oldbalanceDest = 0.0
            newbalanceDest = 0.0

        else:  # tracked_dest_mismatch
            amount = float(np.round(np.random.uniform(100.0, 50000.0), 2))
            oldbalanceOrg = float(np.round(amount + 1000.0, 2))
            newbalanceOrig = float(np.round(oldbalanceOrg - amount, 2))
            oldbalanceDest = 5000.0
            # Tracked destination balance doesn't reflect deposit
            newbalanceDest = float(np.round(np.random.choice([0.0, 5000.0, 5000.0 + amount * 2.0]), 2))

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
    is_cash_in = data["type"] == "CASH_IN"

    # Sender Delta
    data["orgDelta"] = np.where(
        is_cash_in,
        np.round(data["newbalanceOrig"] - data["oldbalanceOrg"] - data["amount"], 2),
        np.round(data["oldbalanceOrg"] - data["newbalanceOrig"] - data["amount"], 2),
    )

    # Destination Delta: if untracked (both 0.0), delta is 0.0
    is_untracked_dest = (data["oldbalanceDest"] == 0.0) & (data["newbalanceDest"] == 0.0)
    data["destDelta"] = np.where(
        is_untracked_dest,
        0.0,
        np.where(
            is_cash_in,
            np.round(data["oldbalanceDest"] - data["newbalanceDest"] - data["amount"], 2),
            np.round(data["newbalanceDest"] - data["oldbalanceDest"] - data["amount"], 2),
        ),
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
        (~is_cash_in) & (data["oldbalanceOrg"] >= 10000) & (data["newbalanceOrig"] == 0), 1, 0
    )

    for t in ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]:
        data[f"type_{t}"] = (data["type"] == t).astype(int)

    return data[FEATURE_COLUMNS]


def train_and_export():
    raw_df = generate_synthetic_transactions(n_samples=32000, fraud_ratio=0.20)
    X = engineer_features(raw_df)
    y = raw_df["isFraud"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)

    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
    )
    gb_model.fit(X_train, y_train)

    rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
    gb_acc = accuracy_score(y_test, gb_model.predict(X_test))
    print(f"RF Acc: {rf_acc*100:.2f}%, GB Acc: {gb_acc*100:.2f}%")

    joblib.dump(rf_model, os.path.join(MODEL_DIR, "rf_model.joblib"))
    joblib.dump(gb_model, os.path.join(MODEL_DIR, "gb_model.joblib"))
    print("Models saved successfully!")


if __name__ == "__main__":
    train_and_export()
