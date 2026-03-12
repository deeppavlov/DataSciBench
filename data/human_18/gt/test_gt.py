import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 1: Perform PCA
# Load data
data = pd.read_csv('../data.csv')

# Preprocessing
data.dropna(inplace=True)
features = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
X = data[features]
y = data['sex']
X_scaled = StandardScaler().fit_transform(X)

# PCA transformation
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Save PCA results
pca_results = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_results['explained_variance_ratio'] = pca.explained_variance_ratio_.sum()
pca_results.to_csv('pca_results.csv', index=False)

# Step 2: Visualize PCA results
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Penguin Dataset')
plt.grid(True)
plt.savefig('pca_visualization.png')

# Step 3: Train a classifier
# Split data
train_X, test_X, train_y, test_y = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# Build classifier
classifier = RandomForestClassifier()
classifier.fit(train_X, train_y)

# Evaluate
predictions = classifier.predict(test_X)
acc = accuracy_score(test_y, predictions)

# Save accuracy
with open('model_accuracy.txt', 'w') as f:
    f.write(f'Classifier Accuracy: {acc:.4f}')