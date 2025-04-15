# scripts/predict_realtime.py
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# Load trained model and scaler
model = load_model('../models/lstm_system_alerts.h5')
scaler = joblib.load('../models/scaler.pkl')

features = ['CPU', 'Memory', 'Load', 'TPS', 'TNS', 'NCTM', 'NEC', 'Disk_Space']
state_map = {0: 'Healthy', 1: 'Amber', 2: 'Red'}

# Example: Load latest data (replace this with actual real-time stream)
df_latest = pd.read_csv('../data/raw/system_metrics_raw.csv').tail(12)  # last 12 entries (1-min history at 5 sec intervals)

# Preprocess data
X_latest = scaler.transform(df_latest[features].values)
X_latest = np.expand_dims(X_latest, axis=0)  # shape: [1, timesteps, features]

# Predict
pred = model.predict(X_latest)
predicted_class = np.argmax(pred, axis=1)[0]

print(f"🚨 Predicted System State: {state_map[predicted_class]}")
