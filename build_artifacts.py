"""
Train the same logistic regression pipeline as the notebook and save artifacts for app.py.
Run once: python build_artifacts.py
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
OUT = Path(__file__).resolve().parent / "model_artifacts.joblib"


def main() -> None:
    df = pd.read_csv(DATA_URL)

    df = df.copy()
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df.drop(columns=["Cabin"], inplace=True, errors="ignore")
    df.drop_duplicates(inplace=True)

    q1 = df["Fare"].quantile(0.25)
    q3 = df["Fare"].quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    df = df[(df["Fare"] >= low) & (df["Fare"] <= high)].copy()

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    df["Title"] = df["Title"].replace(
        ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"],
        "Rare",
    )
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    df["Title"] = df["Title"].fillna("Unknown")

    age_mm_scaler = MinMaxScaler()
    fare_mm_scaler = MinMaxScaler()
    age_std_scaler = StandardScaler()
    fare_std_scaler = StandardScaler()

    df["Age_Normalized"] = age_mm_scaler.fit_transform(df[["Age"]])
    df["Fare_Normalized"] = fare_mm_scaler.fit_transform(df[["Fare"]])
    df["Age_Standardized"] = age_std_scaler.fit_transform(df[["Age"]])
    df["Fare_Standardized"] = fare_std_scaler.fit_transform(df[["Fare"]])

    sex_label_encoder = LabelEncoder()
    df["Sex_enc"] = sex_label_encoder.fit_transform(df["Sex"])

    features = [
        "Pclass",
        "Age_Standardized",
        "Fare_Standardized",
        "SibSp",
        "Parch",
        "FamilySize",
        "IsAlone",
        "Sex_enc",
    ]
    df = df.dropna(subset=features)
    X = df[features]
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(
        {
            "model": model,
            "feature_names": features,
            "age_std_scaler": age_std_scaler,
            "fare_std_scaler": fare_std_scaler,
            "sex_label_encoder": sex_label_encoder,
        },
        OUT,
    )
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
