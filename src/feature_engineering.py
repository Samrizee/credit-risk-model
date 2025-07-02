import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class AggregateFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        agg_features = pd.DataFrame()
        agg_features['Total_Transaction_Amount'] = X.groupby('CustomerId')['Amount'].sum()
        agg_features['Average_Transaction_Amount'] = X.groupby('CustomerId')['Amount'].mean()
        agg_features['Transaction_Count'] = X.groupby('CustomerId')['Amount'].count()
        agg_features['Std_Transaction_Amount'] = X.groupby('CustomerId')['Amount'].std()
        return agg_features.reset_index()

class ExtractFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])
        X['Transaction_Hour'] = X['TransactionStartTime'].dt.hour
        X['Transaction_Day'] = X['TransactionStartTime'].dt.day
        X['Transaction_Month'] = X['TransactionStartTime'].dt.month
        X['Transaction_Year'] = X['TransactionStartTime'].dt.year
        return X