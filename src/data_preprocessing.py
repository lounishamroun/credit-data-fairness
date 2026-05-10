from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo


SEED = 42
AGE_THRESHOLD = 25


@dataclass
class CreditDataSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    age_train: pd.Series
    age_test: pd.Series
    gender_train: pd.Series
    gender_test: pd.Series
    feature_cols: list[str]
    scaler: StandardScaler
    encoders: dict[str, LabelEncoder]


def load_german_credit() -> tuple[pd.DataFrame, pd.Series]:
    """Load the UCI Statlog German Credit dataset."""
    dataset = fetch_ucirepo(id=144)
    X = dataset.data.features.copy()
    y = dataset.data.targets["class"].copy()
    return X, y


def extract_gender(attribute9: pd.Series) -> pd.Series:
    """Map German Credit Attribute9 categories to a binary gender proxy."""
    gender_map = {
        "A91": 0,  # male: divorced/separated
        "A92": 1,  # female: divorced/separated/married
        "A93": 0,  # male: single
        "A94": 0,  # male: married/widowed
        "A95": 1,  # female: single
    }
    return attribute9.map(gender_map).astype(int)


def prepare_credit_data(
    test_size: float = 0.2,
    random_state: int = SEED,
) -> CreditDataSplit:
    """Encode, split, and scale German Credit for the project notebook.

    The target follows the notebook convention:
    - 0: good credit
    - 1: bad credit
    """
    X_raw, y_raw = load_german_credit()
    df = X_raw.copy()

    age_group = (df["Attribute13"] >= AGE_THRESHOLD).astype(int)
    gender = extract_gender(df["Attribute9"])
    target = (y_raw == 2).astype(int)

    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    numerical_cols = df.select_dtypes(include="number").columns.tolist()

    encoders: dict[str, LabelEncoder] = {}
    encoded = df.copy()
    for col in categorical_cols:
        encoder = LabelEncoder()
        encoded[col] = encoder.fit_transform(encoded[col])
        encoders[col] = encoder

    split = train_test_split(
        encoded,
        target,
        age_group,
        gender,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    X_train, X_test, y_train, y_test, age_train, age_test, gender_train, gender_test = split

    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train.loc[:, numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test.loc[:, numerical_cols] = scaler.transform(X_test[numerical_cols])

    return CreditDataSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        age_train=age_train,
        age_test=age_test,
        gender_train=gender_train,
        gender_test=gender_test,
        feature_cols=X_train.columns.tolist(),
        scaler=scaler,
        encoders=encoders,
    )


if __name__ == "__main__":
    prepared = prepare_credit_data()
    print(f"Train shape: {prepared.X_train.shape}")
    print(f"Test shape: {prepared.X_test.shape}")
    print(f"Features: {len(prepared.feature_cols)}")
