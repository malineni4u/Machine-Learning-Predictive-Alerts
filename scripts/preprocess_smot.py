# scripts/preprocess.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE # Import SMOTE
import joblib
import os

# Create directories if they don't exist
os.makedirs('../data/processed', exist_ok=True)
os.makedirs('../models', exist_ok=True)

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

# Encode target variable to numerical labels (required for SMOTE)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y_encoded)

print(f"Original dataset shape: {X.shape}")
print(f"Resampled dataset shape: {X_resampled.shape}")

# Scaling features using the resampled data
# Note: It's generally recommended to fit the scaler *only* on the training data
# after splitting, but since this script preprocesses the entire dataset before
# the split in train_model.py, we fit and transform the resampled data here.
# The scaler should still be saved and used to transform the test set later.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled) # Scale the resampled features

# Save processed data using resampled data
processed_df = pd.DataFrame(X_scaled, columns=features)
# Use the resampled target variable (which is numerically encoded)
# You might want to map it back or save the LabelEncoder mapping if needed elsewhere
processed_df[target] = y_resampled
processed_df.to_csv('../data/processed/system_metrics_processed.csv', index=False)

# Save scaler (fitted on resampled data)
joblib.dump(scaler, '../models/scaler.pkl')
# Optionally save the label encoder if needed later
joblib.dump(le, '../models/label_encoder_preprocess.pkl')

print("✅ Data preprocessing with SMOTE completed.")
print(f"Processed data saved to '../data/processed/system_metrics_processed.csv'")
print(f"Scaler saved to '../models/scaler.pkl'")