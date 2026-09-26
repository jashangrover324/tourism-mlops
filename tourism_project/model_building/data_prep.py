# data_prep.py
# Loads the raw tourism dataset, cleans it, engineers features, and produces
# stratified train/test splits ready for model training.

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

RAW_PATH = "tourism_project/data/tourism.csv"
OUT_DIR = "tourism_project/data"

TARGET = "ProdTaken"
ID_COLS = ["Unnamed: 0", "CustomerID"]

def load_data(path=RAW_PATH):
    df = pd.read_csv(path)
    # Drop the stray index column written by pandas on export, if present
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    return df

def clean_data(df):
    df = df.copy()

    # Fix inconsistent category labels found during EDA
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    if "MaritalStatus" in df.columns:
        df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

    # Drop the customer identifier - it carries no predictive signal
    df = df.drop(columns=[c for c in ID_COLS if c in df.columns], errors="ignore")

    # Impute numeric missing values with the median (robust to outliers)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET in num_cols:
        num_cols.remove(TARGET)
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Impute categorical missing values with the mode
    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    return df

def split_data(df, test_size=0.2, random_state=42):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_data()
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df)

    X_train.to_csv(f"{OUT_DIR}/X_train.csv", index=False)
    X_test.to_csv(f"{OUT_DIR}/X_test.csv", index=False)
    y_train.to_csv(f"{OUT_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{OUT_DIR}/y_test.csv", index=False)

    print("Data preparation complete.")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Train positive rate: {y_train.mean():.3f}, Test positive rate: {y_test.mean():.3f}")

if __name__ == "__main__":
    main()
