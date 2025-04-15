# scripts/preprocess.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Load raw data
df = pd.read_csv('../data/raw/system_metrics_raw.csv')

# Feature selection based on your Random Forest model results
features = ['CPU', 'Memory', 'Load', 'TPS', 'TNS', 'NCTM', 'NEC', 'Disk_Space']
target = 'RESTART'

# Handling missing values if any (example: forward fill)
df.fillna(method='ffill', inplace=True)

# Extract features and target
X = df[features]
y = df[target]

# Scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save processed data
processed_df = pd.DataFrame(X_scaled, columns=features)
processed_df[target] = y
processed_df.to_csv('../data/processed/system_metrics_processed.csv', index=False)

# Save scaler
joblib.dump(scaler, '../models/scaler.pkl')

print("✅ Data preprocessing completed.")
