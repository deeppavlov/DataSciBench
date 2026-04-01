import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# Load input and output datasets
input_data = pd.read_csv("../input_data.csv").values
output_data = pd.read_csv("../output_data.csv").values

# Transpose output_data matrix so shapes match for supervised learning
output_data = output_data.T

# Ensure the input and output shapes are correct
assert input_data.shape == (500, 12)
assert output_data.shape == (500, 7)

# Build a simple neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(12,)),  # Input layer with 12 nodes
    Dense(64, activation='relu'),                     # Hidden layer with 64 nodes
    Dense(7, activation='linear')                     # Output layer with 7 nodes
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
history = model.fit(input_data, output_data, epochs=100, batch_size=32, validation_split=0.2)

# Save the model to a file "model.h5"
model.save("model.h5")

# Evaluate the model
evaluation_results = model.evaluate(input_data, output_data, verbose=0)

# Save evaluation results to a CSV file
eval_df = pd.DataFrame([evaluation_results], columns=['Loss', 'MeanAbsoluteError'])
eval_df.to_csv("evaluation_results.csv", index=False)

print("Training complete. Model saved to 'model.h5' and evaluation results saved to 'evaluation_results.csv'.")