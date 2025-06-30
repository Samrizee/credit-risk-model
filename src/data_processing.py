import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Custom Transformer: Extract time features
class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, date_col):
        self.date_col = date_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df["transaction_hour"] = df[self.date_col].dt.hour
        df["transaction_day"] = df[self.date_col].dt.day
        df["transaction_month"] = df[self.date_col].dt.month
        df["transaction_year"] = df[self.date_col].dt.year
        return df.drop(columns=[self.date_col])

# Custom Transformer: Aggregate features
class AggregateFeatureCreator(BaseEstimator, TransformerMixin):
    def __init__(self, group_col, value_col):
        self.group_col = group_col
        self.value_col = value_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        agg = X.groupby(self.group_col)[self.value_col].agg([
            ("total_transaction_amount", "sum"),
            ("avg_transaction_amount", "mean"),
            ("transaction_count", "count"),
            ("std_transaction_amount", "std")
        ]).reset_index()
        return agg

def build_pipeline(date_col, group_col, value_col, cat_cols, num_cols):
    # Feature extraction steps
    date_features = DateFeatureExtractor(date_col=date_col)
    aggregate_features = AggregateFeatureCreator(group_col=group_col, value_col=value_col)

    # Preprocessing
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="mean")),
        ("scale", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols)
    ])

    # Final pipeline
    pipeline = Pipeline([
        ("extract_date", date_features),
        ("aggregate", aggregate_features),
        ("preprocess", preprocessor)
    ])

    return pipeline
