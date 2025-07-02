

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def compute_rfm(df):
    df = df.copy()
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
        'TransactionId': 'count',
        'Amount': 'sum'
    }).rename(columns={
        'TransactionStartTime': 'Recency',
        'TransactionId': 'Frequency',
        'Amount': 'Monetary'
    }).reset_index()

    return rfm

def cluster_customers(rfm, n_clusters=3, random_state=42):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    rfm['cluster'] = kmeans.fit_predict(rfm_scaled)

    # Define high-risk cluster based on lowest Frequency
    high_risk_cluster = rfm.groupby('cluster')['Frequency'].mean().idxmin()
    rfm['is_high_risk'] = (rfm['cluster'] == high_risk_cluster).astype(int)

    return rfm[['CustomerId', 'is_high_risk']]

def add_high_risk_label(processed_df, df):
    rfm = compute_rfm(df)
    risk_labels = cluster_customers(rfm)
    final_data = processed_df.merge(risk_labels, on='CustomerId', how='left')
    return final_data