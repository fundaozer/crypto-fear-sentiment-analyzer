import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def prepare_features(df):
    """
    Creates target variable 'next_day_down' (1 if next_day_return < 0, else 0),
    selects required features, drops NaN rows, and returns X and y.

    Haber olmayan günlerde (headline_count=0 veya NaN) sentiment_negativity ve
    sentiment_negativity_scaled NaN olur. Bu sütunlar ile market_stress_index,
    dropna'dan önce 0 ile doldurulur.
    """
    df = df.copy()
    
    # Target: 1 if next day price drops, 0 otherwise.
    # Set to NaN if next_day_return is NaN (e.g. for the last row) so it gets dropped.
    df["next_day_down"] = np.where(
        df["next_day_return"].isna(),
        np.nan,
        np.where(df["next_day_return"] < 0, 1, 0)
    )

    features = [
        "volatility",
        "volume_spike",
        "sentiment_negativity",
        "fear_greed_value",
        "market_stress_index",
        "panic_score",
        "momentum_score"
    ]

    # Haber olmayan günlerde bu sütunlar NaN gelir; 0 (nötr) kabul edilir.
    df["sentiment_negativity"] = df["sentiment_negativity"].fillna(0)
    df["sentiment_negativity_scaled"] = df["sentiment_negativity_scaled"].fillna(0)
    df["market_stress_index"] = df["market_stress_index"].fillna(0)

    # Drop rows with NaN in features or target
    clean_df = df.dropna(subset=features + ["next_day_down"])

    X = clean_df[features]
    y = clean_df["next_day_down"].astype(int)

    return X, y

def split_data(X, y):
    """
    Performs chronological/sequential train-test split (first 80% train, last 20% test).
    """
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train, y_train):
    """
    Trains a Logistic Regression model within a pipeline that includes StandardScaler.
    """
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=42)
    )
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """
    Trains a Random Forest Classifier model.
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, model_name):
    """
    Predicts labels on X_test and returns dictionary of metrics.
    """
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    results = {
        "model_name": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm.tolist()
    }
    return results
