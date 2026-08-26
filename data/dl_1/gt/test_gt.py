import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

INPUT_SIZE = 7
HIDDEN_SIZE = 10
OUTPUT_SIZE = 1
POP_SIZE = 20
SSA_ITERS = 30
BP_EPOCHS = 4000
LEARNING_RATE = 0.05
SEED = 42

data = pd.read_excel('../data.xlsx')
scaler = StandardScaler()
scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
X = scaled.drop('output', axis=1).values
y = scaled['output'].values.reshape(-1, 1)
idx = np.arange(len(y))
train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=SEED)
X_train, y_train = X[train_idx], y[train_idx]

n_ih = INPUT_SIZE * HIDDEN_SIZE
n_ho = HIDDEN_SIZE * OUTPUT_SIZE
DIM = n_ih + n_ho + HIDDEN_SIZE + OUTPUT_SIZE


def unpack(vector):
    w1 = vector[:n_ih].reshape(INPUT_SIZE, HIDDEN_SIZE)
    w2 = vector[n_ih:n_ih + n_ho].reshape(HIDDEN_SIZE, OUTPUT_SIZE)
    b1 = vector[n_ih + n_ho:n_ih + n_ho + HIDDEN_SIZE]
    b2 = vector[-OUTPUT_SIZE:]
    return w1, w2, b1, b2


def forward(vector, x):
    w1, w2, b1, b2 = unpack(vector)
    hidden = np.tanh(x @ w1 + b1)
    return hidden @ w2 + b2


def fitness(vector):
    return float(np.mean((forward(vector, X_train) - y_train) ** 2))


rng = np.random.default_rng(SEED)
population = rng.uniform(-1, 1, (POP_SIZE, DIM))
scores = np.array([fitness(p) for p in population])
best = population[scores.argmin()].copy()
best_score = scores.min()
producers = POP_SIZE // 5

for _ in range(SSA_ITERS):
    order = scores.argsort()
    population, scores = population[order], scores[order]
    worst = population[-1]
    alarm = rng.random()
    for i in range(POP_SIZE):
        if i < producers:
            if alarm < 0.8:
                population[i] = population[i] * np.exp(-(i + 1) / (0.01 + rng.random()) / SSA_ITERS)
            else:
                population[i] = population[i] + rng.normal(size=DIM)
        elif i > POP_SIZE // 2:
            population[i] = rng.normal(size=DIM) * np.exp((worst - population[i]) / (i + 1) ** 2)
        else:
            population[i] = population[0] + np.abs(population[i] - population[0]) * rng.choice([-1.0, 1.0], DIM)
    population = np.clip(population, -3, 3)
    scores = np.array([fitness(p) for p in population])
    if scores.min() < best_score:
        best_score = scores.min()
        best = population[scores.argmin()].copy()

w1, w2, b1, b2 = [a.copy() for a in unpack(best)]
first_moment = [np.zeros_like(a) for a in (w1, w2, b1, b2)]
second_moment = [np.zeros_like(a) for a in (w1, w2, b1, b2)]
n_train = len(X_train)

for epoch in range(1, BP_EPOCHS + 1):
    hidden = np.tanh(X_train @ w1 + b1)
    error = hidden @ w2 + b2 - y_train
    grad_w2 = hidden.T @ error * 2 / n_train
    grad_b2 = error.sum(axis=0) * 2 / n_train
    delta = (error @ w2.T) * (1 - hidden ** 2)
    grad_w1 = X_train.T @ delta * 2 / n_train
    grad_b1 = delta.sum(axis=0) * 2 / n_train
    grads = (grad_w1, grad_w2, grad_b1, grad_b2)
    for param, grad, m, v in zip((w1, w2, b1, b2), grads, first_moment, second_moment):
        m *= 0.9
        m += 0.1 * grad
        v *= 0.999
        v += 0.001 * grad * grad
        param -= LEARNING_RATE * (m / (1 - 0.9 ** epoch)) / (np.sqrt(v / (1 - 0.999 ** epoch)) + 1e-8)

params = np.concatenate([w1.ravel(), w2.ravel(), b1.ravel(), b2.ravel()])
np.savetxt('model_parameters.csv', params, delimiter=',')

predictions = forward(params, X).ravel() * scaler.scale_[-1] + scaler.mean_[-1]
pd.DataFrame({'Predicted': predictions}).to_csv('predictions.csv', index=False)

actual = data['output'].values
y_test = actual[test_idx]
y_pred_test = predictions[test_idx]
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae = mean_absolute_error(y_test, y_pred_test)
mape = mean_absolute_percentage_error(y_test, y_pred_test) * 100
r2 = r2_score(y_test, y_pred_test)
with open('performance_metrics.txt', 'w') as f:
    f.write(f"RMSE: {rmse}\nMAE: {mae}\nMAPE: {mape}\nR2: {r2}\n")

print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"MAPE: {mape}")
print(f"R2: {r2}")
print(f"Full-data MAE: {mean_absolute_error(actual, predictions)}")
print(f"Full-data R2: {r2_score(actual, predictions)}")
