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

import joblib
import os

# Create directories if they don't exist
os.makedirs('../models', exist_ok=True)
os.makedirs('../visuals', exist_ok=True)

# Load processed data
df = pd.read_csv('../data/processed/system_metrics_processed.csv')

features = ['CPU', 'Memory', 'Load', 'TPS', 'TNS', 'NCTM', 'NEC', 'Disk_Space']
target = 'RESTART'

X = df[features].values
y, labels = pd.factorize(df[target])  # Encoding target labels as integers

# Save the label encoder mapping
joblib.dump(labels, '../models/label_encoder.pkl')

# Convert target to categorical
y_cat = to_categorical(y, num_classes=len(labels))

# Prepare sequences
def create_sequences(X, y, time_steps=12):
    X_seqs, y_seqs = [], []
    for i in range(len(X) - time_steps):
        X_seqs.append(X[i:(i+time_steps)])
        y_seqs.append(y[i+time_steps])
    return np.array(X_seqs), np.array(y_seqs)

X_seq, y_seq = create_sequences(X, y_cat)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# Model definition
model = Sequential([
    LSTM(64, activation='relu', return_sequences=True, input_shape=(X_seq.shape[1], X_seq.shape[2])),
    Dropout(0.2),
    LSTM(32, activation='relu'),
    Dropout(0.2),
    Dense(len(labels), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Train model
#history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history =model.fit(X_train, y_train, epochs=100, batch_size=32, 
          validation_data=(X_test, y_test), callbacks=[early_stop])

# Save trained model
model.save('../models/lstm_system_alerts.h5')

# Plot training & validation accuracy and loss
plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('../visuals/performance_plot.png')

# Predict on test data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Confusion Matrix visualization
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.savefig('../visuals/confusion_matrix.png')

# Classification Report
report = classification_report(y_true_classes, y_pred_classes, target_names=labels)
with open('../visuals/classification_report.txt', 'w') as f:
    f.write(report)

print("✅ Model trained, saved, and visuals created successfully.")
