from __future__ import annotations
from typing import List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xverse.transformer import WOETransformer  # Optional dependency
except ModuleNotFoundError:
    WOETransformer = None  # Allow pipeline to run without it

# -----------------------------------------------------------------------------
# 🔧 Constants

ID_COL: str = "CustomerId"
DATETIME_COL: str = "TransactionStartTime"
AMOUNT_COL: str = "Amount"
TARGET: str = "FraudResult"  # Not used inside transformations, but exposed

AGG_SUFFIXES: List[str] = ["_sum", "_mean", "_count", "_std"]

# -----------------------------------------------------------------------------
# 📦 Transformers

class Aggregator(BaseEstimator, TransformerMixin):
    def __init__(self, id_col: str = ID_COL, amount_col: str = AMOUNT_COL):
        self.id_col = id_col
        self.amount_col = amount_col
        self._feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        aggs = (
            X.groupby(self.id_col)[self.amount_col]
            .agg(["sum", "mean", "count", "std"])
            .rename(columns={
                "sum": f"{self.amount_col}_sum",
                "mean": f"{self.amount_col}_mean",
                "count": f"{self.amount_col}_count",
                "std": f"{self.amount_col}_std",
            })
        )
        self._aggregates_ = aggs
        self._feature_names = aggs.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "_aggregates_"):
            raise RuntimeError("Must call fit before transform on Aggregator")
        return X.merge(self._aggregates_, how="left", left_on=self.id_col, right_index=True)

    def get_feature_names_out(self, input_features: list[str] | None = None):
        return np.array(self._feature_names)

class DatetimeExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_col: str = DATETIME_COL):
        self.datetime_col = datetime_col
        self._feature_names: list[str] = [
            f"{self.datetime_col}_hour",
            f"{self.datetime_col}_day",
            f"{self.datetime_col}_month",
            f"{self.datetime_col}_year",
        ]

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        dt = pd.to_datetime(X[self.datetime_col], errors="coerce")
        new_cols = pd.DataFrame({
            f"{self.datetime_col}_hour": dt.dt.hour.astype("Int16"),
            f"{self.datetime_col}_day": dt.dt.day.astype("Int16"),
            f"{self.datetime_col}_month": dt.dt.month.astype("Int16"),
            f"{self.datetime_col}_year": dt.dt.year.astype("Int16"),
        }, index=X.index)
        return pd.concat([X, new_cols], axis=1)

    def get_feature_names_out(self, input_features: list[str] | None = None):
        return np.array(self._feature_names)

# -----------------------------------------------------------------------------
# 🔁 Build Pipeline

def build_pipeline() -> Pipeline:
    numeric_features = [
        f"{AMOUNT_COL}_sum",
        f"{AMOUNT_COL}_mean",
        f"{AMOUNT_COL}_count",
        f"{AMOUNT_COL}_std",
    ]
    datetime_features = [
        f"{DATETIME_COL}_hour",
        f"{DATETIME_COL}_day",
        f"{DATETIME_COL}_month",
        f"{DATETIME_COL}_year",
    ]
    categorical_features = [
        "CurrencyCode",
        "ChannelId",
        "ProviderId",
        "ProductId",
        "ProductCategory"

    ]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]), numeric_features + datetime_features),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_features),
    ])

    steps: list[tuple[str, BaseEstimator]] = [
        ("extract", DatetimeExtractor(datetime_col=DATETIME_COL)),
        ("aggregate", Aggregator(id_col=ID_COL, amount_col=AMOUNT_COL)),
        ("prep", preprocessor),
    ]

    if WOETransformer is not None:
        steps.append(("woe", WOETransformer(features_to_encode="auto")))

    return Pipeline(steps)

# -----------------------------------------------------------------------------
# 📌 Feature Set Definitions

AGG_COLS = [f"{AMOUNT_COL}{suf}" for suf in AGG_SUFFIXES]
FEATURES: List[str] = [ID_COL, DATETIME_COL, AMOUNT_COL, *AGG_COLS]
