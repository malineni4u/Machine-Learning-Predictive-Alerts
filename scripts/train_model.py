# scripts/train_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers

import joblib # Make sure joblib is imported
import os

def create_sequences(X, y, time_steps=12):
    X_seqs, y_seqs = [], []
    for i in range(len(X) - time_steps):
        X_seqs.append(X[i:(i+time_steps)])
        y_seqs.append(y[i+time_steps])
    return np.array(X_seqs), np.array(y_seqs)

# Load processed data (Now contains numerical target from SMOTE preprocessing)
df = pd.read_csv('../data/processed/system_metrics_processed.csv')

features = ['CPU', 'Memory', 'Load', 'TPS', 'TNS', 'NCTM', 'NEC', 'Disk_Space']
target = 'RESTART'

X = df[features].values
# The target column 'RESTART' is already numerical (0, 1, 2...)
# We still use factorize to get the y array in the right format,
# but we ignore its second return value (labels) as it's just the numbers now.
y, _ = pd.factorize(df[target]) # <--- CHANGE: Ignore the second return value

# Load the *original* class labels from the encoder saved during preprocessing
try:
    le = joblib.load('../models/label_encoder_preprocess.pkl') # Load the correct encoder
    original_labels = le.classes_ # Get the original string labels ('Healthy', 'Amber', 'Red')
    num_classes = len(original_labels)
except FileNotFoundError:
    print("Error: label_encoder_preprocess.pkl not found. Please run the updated preprocess.py script first.")
    exit()
except Exception as e:
    print(f"Error loading label encoder: {e}")
    exit()

# --- (Keep code for converting y to categorical, creating sequences, train-test split) ---
# Ensure num_classes is correctly determined
y_cat = to_categorical(y, num_classes=num_classes) # Use num_classes from loaded labels

# ... (rest of the sequence creation and train-test split) ...
X_seq, y_seq = create_sequences(X, y_cat, time_steps=6) # Assuming X is already scaled
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# --- (Keep Model definition, compilation, training) ---
# Ensure the final Dense layer uses the correct number of classes
model = Sequential([
    LSTM(16, activation='relu', return_sequences=True, input_shape=(X_seq.shape[1], X_seq.shape[2]),kernel_regularizer=regularizers.l2(0.0001)),
    Dropout(0.5),
    LSTM(8, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax') #<--- Ensure this uses num_classes
])

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0005), metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                    validation_data=(X_test, y_test), callbacks=[early_stop])

# --- (Keep code for saving model, plotting performance) ---
model.save('../models/lstm_system_alerts.h5')
# ... (performance plotting code) ...

# Predict on test data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1) # These are numerical indices (0, 1, 2...)

# --- Evaluation ---
# Confusion Matrix visualization using original labels
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(8,6))
# Use the loaded original_labels for heatmap ticks
sns.heatmap(cm, annot=True, fmt='d', xticklabels=original_labels, yticklabels=original_labels, cmap='Blues') # <--- CHANGE
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.savefig('../visuals/confusion_matrix.png')

# Classification Report using original labels
# Use the loaded original_labels for target_names
report = classification_report(y_true_classes, y_pred_classes, target_names=original_labels) # <--- CHANGE
with open('../visuals/classification_report.txt', 'w') as f:
    f.write(report)

print("✅ Model trained, saved, and visuals created successfully.")
# Optional: Remove the redundant saving of the incorrect labels
# joblib.dump(labels, '../models/label_encoder.pkl') # REMOVE or comment out this line