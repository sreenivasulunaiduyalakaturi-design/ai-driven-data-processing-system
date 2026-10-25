"""
AI-Driven Data Processing System
--------------------------------
A simple base pipeline:
  1. Load      -> read CSV (or generate sample data)
  2. Clean     -> remove duplicates, fix types, impute missing values
  3. Transform -> encode categoricals, scale numerics
  4. AI        -> anomaly detection, clustering, optional prediction model
  5. Report    -> save processed CSV + JSON summary

Usage:
    python main.py                              # runs on generated sample data
    python main.py --input data.csv             # your own data
    python main.py --input data.csv --target price   # also trains a model
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------- 1. LOAD
def load_data(path=None):
    if path:
        print(f"[load] Reading {path}")
        return pd.read_csv(path)
    print("[load] No input given -> generating sample data")
    rng = np.random.default_rng(42)
    n = 500
    df = pd.DataFrame({
        "age": rng.integers(18, 70, n).astype(float),
        "income": rng.normal(50000, 15000, n),
        "spending_score": rng.integers(1, 100, n).astype(float),
        "city": rng.choice(["Hyderabad", "Mumbai", "Delhi", "Chennai"], n),
    })
    df.loc[rng.choice(n, 10, replace=False), "income"] = rng.uniform(250000, 400000, 10)  # outliers
    df.loc[rng.choice(n, 25, replace=False), "age"] = np.nan                               # missing
    df.loc[rng.choice(n, 15, replace=False), "city"] = np.nan
    return pd.concat([df, df.sample(10, random_state=1)], ignore_index=True)               # duplicates


# --------------------------------------------------------------- 2. CLEAN
def clean_data(df):
    report = {"rows_before": len(df)}
    df = df.drop_duplicates().copy()
    report["duplicates_removed"] = report["rows_before"] - len(df)

    num_cols = df.select_dtypes(include="number").columns
    cat_cols = df.select_dtypes(exclude="number").columns
    report["missing_values_filled"] = int(df.isna().sum().sum())

    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        mode = df[c].mode()
        df[c] = df[c].fillna(mode[0] if not mode.empty else "unknown")

    report["rows_after"] = len(df)
    print(f"[clean] {report}")
    return df.reset_index(drop=True), report


# ----------------------------------------------------------- 3. TRANSFORM
def transform_data(df, target=None):
    """Return a fully numeric, scaled feature matrix (target excluded)."""
    features = df.drop(columns=[target]) if target else df.copy()
    features = pd.get_dummies(features, drop_first=True)
    scaled = StandardScaler().fit_transform(features)
    print(f"[transform] Feature matrix: {scaled.shape}")
    return pd.DataFrame(scaled, columns=features.columns, index=df.index)


# ------------------------------------------------------------------ 4. AI
def detect_anomalies(X, contamination=0.03):
    model = IsolationForest(contamination=contamination, random_state=42)
    return model.fit_predict(X) == -1  # True = anomaly


def find_clusters(X, k=3):
    return KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)


def train_model(X, y):
    is_classification = y.dtype == object or y.nunique() <= 10
    model = RandomForestClassifier(random_state=42) if is_classification else RandomForestRegressor(random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    if is_classification:
        metric = {"task": "classification", "accuracy": round(accuracy_score(y_te, pred), 4)}
    else:
        metric = {"task": "regression", "r2": round(r2_score(y_te, pred), 4)}
    top = pd.Series(model.feature_importances_, index=X.columns).nlargest(5)
    metric["top_features"] = {k: round(v, 4) for k, v in top.items()}
    return metric


# ---------------------------------------------------------------- 5. REPORT
def save_outputs(df, summary, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "processed_data.csv"), index=False)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[report] Saved results to '{out_dir}/'")


# ------------------------------------------------------------------- MAIN
def main():
    p = argparse.ArgumentParser(description="AI-Driven Data Processing System")
    p.add_argument("--input", help="Path to input CSV")
    p.add_argument("--target", help="Optional column to predict")
    p.add_argument("--clusters", type=int, default=3)
    p.add_argument("--output", default="output")
    args = p.parse_args()

    df = load_data(args.input)
    df, clean_report = clean_data(df)
    X = transform_data(df, args.target)

    df["is_anomaly"] = detect_anomalies(X)
    df["cluster"] = find_clusters(X, args.clusters)

    summary = {
        "cleaning": clean_report,
        "anomalies_found": int(df["is_anomaly"].sum()),
        "cluster_sizes": df["cluster"].value_counts().sort_index().to_dict(),
    }
    if args.target:
        summary["model"] = train_model(X, df[args.target])

    print(f"[ai] Anomalies: {summary['anomalies_found']} | Clusters: {summary['cluster_sizes']}")
    if "model" in summary:
        print(f"[ai] Model: {summary['model']}")
    save_outputs(df, summary, args.output)


if __name__ == "__main__":
    main()
