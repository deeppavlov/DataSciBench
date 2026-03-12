import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import joblib

# Load input data
input_data = pd.read_csv('../12_Input.csv')
output_data = pd.read_csv('../a_full.csv')

# Transposing the output_data to match corresponding input-output pairs
output_data = output_data.T

# Assume each row of the input predicts a corresponding column of the output
# Use the first row as an example
X = input_data.values
Y = output_data.values.T  # Transpose to align with X for row-wise prediction

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create a model for each output column and fit
models = []
predictions = []
actual = []

for i in range(Y_train.shape[1]):
    model = SVR(kernel='rbf')
    model.fit(X_train, Y_train[:, i])
    models.append(model)
    prediction = model.predict(X_test)
    predictions.append(prediction)
    actual.append(Y_test[:, i])

# Combine predictions and actual values for accuracy visualization
predictions = np.array(predictions)
actual = np.array(actual)

# Calculate prediction accuracy
r2_scores = [r2_score(actual[i], predictions[i]) for i in range(predictions.shape[0])]
mse_scores = [mean_squared_error(actual[i], predictions[i]) for i in range(predictions.shape[0])]

# Save the model
joblib.dump(models, 'model.pkl')

# Plot prediction accuracy
plt.figure(figsize=(10, 6))
plt.plot(range(len(r2_scores)), r2_scores, label='R2 Score')
plt.plot(range(len(mse_scores)), mse_scores, label='MSE')
plt.xlabel('Column Index')
plt.ylabel('Score')
plt.title('Prediction Accuracy (R2 Score and MSE) for each Output Column')
plt.legend()
plt.savefig('prediction_accuracy.png')

# Print an example prediction vs actual
print("Example prediction for the first row of input:")
example_input = X_test[0].reshape(1, -1)
example_prediction = [models[i].predict(example_input)[0] for i in range(len(models))]
print("Predicted:", example_prediction)
print("Actual:", Y_test[0])
