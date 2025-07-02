from sklearn.pipeline import Pipeline
import pandas as pd
from feature_engineering import AggregateFeatures, ExtractFeatures
from preprocessing import create_preprocessing_pipeline

def main(data):
    # Clean column names
    data.columns = data.columns.str.strip()

    # Define required input features
    categorical_features = ['ProviderId', 'ProductId', 'ProductCategory', 'ChannelId', 'PricingStrategy']
    numerical_features = ['Amount']
    date_features = ['TransactionStartTime']

    # Ensure all required input columns are present
    required_cols = categorical_features + numerical_features + date_features
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    print("Initial columns:", data.columns.tolist())

    # 🔁 Step 1: Feature Engineering (must come before preprocessing!)
    feature_pipeline = Pipeline(steps=[
        ('extract_features', ExtractFeatures()),
        ('aggregate_features', AggregateFeatures())
    ])
    engineered_data = feature_pipeline.fit_transform(data)
    print("Engineered columns:", engineered_data.columns.tolist())

    # 🔁 Step 2: Preprocessing (on the engineered output)
    # ⚠️ Update this to match what columns exist after feature engineering
    categorical_features = []  # If no categorical left
    numerical_features = [col for col in engineered_data.columns if engineered_data[col].dtype in ['float64', 'int64'] and col != 'CustomerId']

    preprocessor = create_preprocessing_pipeline(categorical_features, numerical_features)
    processed_data = preprocessor.fit_transform(engineered_data)

    # Optional: wrap back into DataFrame
    processed_df = pd.DataFrame(processed_data)

    print("Final processed data shape:", processed_df.shape)
    return processed_df
