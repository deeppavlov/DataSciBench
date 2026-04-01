import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

# Constants
INPUT_DIR = '../data/'
OUTPUT_GRAPH_STRUCTURE_FILE = 'graph_structure.json'
OUTPUT_COMBINED_DATA_FILE = 'combined_data.csv'
OUTPUT_SEGMENTED_DATA_FILE = 'segmented_data.csv'
TRAIN_DATA_FILE = 'train_data.csv'
TEST_DATA_FILE = 'test_data.csv'
TRAINED_MODEL_FILE = 'trained_model.h5'
TRAINING_METRICS_FILE = 'training_metrics.csv'
CLASSIFICATION_REPORT_FILE = 'classification_report.txt'
CONFUSION_MATRIX_FILE = 'confusion_matrix.png'

# Load data from excel files
excel_files = glob.glob(os.path.join(INPUT_DIR, 'data_*.xlsx'))

# Combine data from all Excel files
combined_data = []
for file in excel_files:
    xls = pd.ExcelFile(file)
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(file, sheet_name=sheet_name)
        combined_data.append(df)

# Combine into one DataFrame
combined_df = pd.concat(combined_data, ignore_index=True)
combined_df.to_csv(OUTPUT_COMBINED_DATA_FILE, index=False)

# Segment the data using sliding window
l = 50  # window length
s = 10  # step size

def segment_data(data, window_length, step_size):
    segments = []
    for start in range(0, len(data) - window_length, step_size):
        end = start + window_length
        segment = data.iloc[start:end]
        segments.append(segment)
    return pd.concat(segments, ignore_index=True)

segmented_df = segment_data(combined_df, l, s)
segmented_df.to_csv(OUTPUT_SEGMENTED_DATA_FILE, index=False)

# Split into training and testing sets
train_data, test_data = train_test_split(segmented_df, test_size=0.3, random_state=42)
train_data.to_csv(TRAIN_DATA_FILE, index=False)
test_data.to_csv(TEST_DATA_FILE, index=False)

# Prepare the data for GRNN model
def prepare_data(df):
    X = df.drop(columns=['label']).values
    y = df['label'].values
    X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for LSTM [samples, time steps, features]
    y = tf.keras.utils.to_categorical(y)  # One-hot encode labels
    return X, y

X_train, y_train = prepare_data(train_data)
X_test, y_test = prepare_data(test_data)

# Create and train the GRNN model
model = Sequential()
model.add(LSTM(100, input_shape=(l, 1), return_sequences=True))
model.add(LSTM(50))
model.add(Dense(y_train.shape[1], activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model
history = model.fit(X_train, y_train, epochs=60, batch_size=64, validation_split=0.2)

# Save the trained model
model.save(TRAINED_MODEL_FILE)

# Log training metrics
metrics = pd.DataFrame(history.history)
metrics.to_csv(TRAINING_METRICS_FILE, index=False)

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Classification report
report = classification_report(y_true_classes, y_pred_classes, output_dict=True)
with open(CLASSIFICATION_REPORT_FILE, 'w') as f:
    f.write(classification_report(y_true_classes, y_pred_classes))

# Confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.savefig(CONFUSION_MATRIX_FILE)

# Graph structure representation
graph_structure = {
    "nodes": ["Vehicle_" + str(i) for i in range(1, len(combined_df.columns))],
    "edges": [
        {"from": "Vehicle_" + str(i), "to": "Vehicle_" + str(j)}
        for i in range(1, len(combined_df.columns))
        for j in range(1, len(combined_df.columns)) if i != j
    ]
}
with open(OUTPUT_GRAPH_STRUCTURE_FILE, 'w') as f:
    json.dump(graph_structure, f, indent=4)

print("Process complete. All outputs have been saved.")