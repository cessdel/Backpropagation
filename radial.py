import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def initialize_centroids(X, K):
    indices = np.random.choice(len(X), K, replace=False)
    return X[indices]

def assign_clusters(X, centroids):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(X, labels, K):
    return np.array([X[labels == k].mean(axis=0) for k in range(K)])
def compute_dynamically_d(centroids):
    K = len(centroids)
    dist_sum = sum(np.linalg.norm(centroids[i] - centroids[j]) 
                    for i in range(K) for j in range(i + 1, K))
    count = K * (K - 1) / 2
    return dist_sum / count if count > 0 else 1.0

def kmeans(X, K, tol=1e-4, max_iter=200):
    centroids = initialize_centroids(X, K)
    for _ in range(max_iter):
        old_centroids = centroids.copy()
        labels = assign_clusters(X, centroids)
        centroids = update_centroids(X, labels, K)
        if np.linalg.norm(centroids - old_centroids) < tol:
            break
    return centroids, labels

def gaussian_function(X, centers, d):
    r = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
    return np.exp(- (r ** 2) / (2 * d ** 2))

def initialize_weights(input_size, output_size):
    return np.random.uniform(-1, 1, size=(input_size, output_size)), np.random.uniform(-1, 1, size=(1, output_size))

def train_rbf(X_train, y_train, centers, d, alpha, epochs, tol=1e-6, patience=50):
    M = centers.shape[0]  # Número de neuronas en la capa intermedia (igual a las clases) (3, 4)
    K = y_train.shape[1]  # Número de neuronas en la capa de salida (120, 3)
    W, W0 = initialize_weights(M, K)
    epoch_errors = []
    best_error = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        total_error = 0
        for X, t in zip(X_train, y_train):
            phi = gaussian_function(X.reshape(1, -1), centers, d)
            Y = phi @ W + W0
            error = t - Y
            W += alpha * error * phi.T
            W0 += alpha * error
            total_error += np.mean(np.abs(error))
        
        avg_error = total_error / len(X_train)
        epoch_errors.append(avg_error)
        
        if avg_error < best_error - tol:
            best_error = avg_error
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    return W, W0, epoch_errors

def predict_rbf(X, centers, d, W, W0):
    phi = gaussian_function(X, centers, d)
    Y = phi @ W + W0
    return np.argmax(Y, axis=1)

# Cargar dataset
iris = datasets.load_iris()
X = iris.data
Y = iris.target.reshape(-1, 1)
encoder = OneHotEncoder(sparse_output=False)
Y_one_hot = encoder.fit_transform(Y)

# Parámetros
K = 3
alpha = 0.05
epochs = 10000
repetitions = 35
total_epoch_errors = []
metrics = []

for i in range(repetitions):
    X_train, X_test, y_train, y_test = train_test_split(X, Y_one_hot, test_size=0.2, random_state=42)
    final_centroids, labels = kmeans(X_train, K)
    d = compute_dynamically_d(final_centroids)
    W, W0, epoch_errors = train_rbf(X_train, y_train, final_centroids, d, alpha, epochs)
    total_epoch_errors.append(epoch_errors)
    print(f"Valor de D: {d}")
    y_pred = predict_rbf(X_test, final_centroids, d, W, W0)
    y_test_labels = np.argmax(y_test, axis=1)
    accuracy = np.mean(y_pred == y_test_labels)
    precision = np.mean(y_pred[y_test_labels == 1] == 1) if np.any(y_test_labels == 1) else 0
    recall = np.mean(y_pred == y_test_labels)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    metrics.append((accuracy, precision, recall, f1_score))

# Graficar errores por época en cada experimento
plt.figure(figsize=(10, 5))
for errors in total_epoch_errors:
    plt.plot(range(1, len(errors) + 1), errors, linestyle='-', alpha=0.5)
plt.xlabel('Época')
plt.ylabel('Error de Clasificación')
plt.title('Evolución del Error en 35 Repeticiones')
plt.show()

# Imprimir métricas
accuracies, precisions, recalls, f1_scores = zip(*metrics)
print(f"Promedio de exactitud: {np.mean(accuracies):.4f}")
print(f"Promedio de precisión (Precision): {np.mean(precisions):.4f}")
print(f"Promedio de recall: {np.mean(recalls):.4f}")
print(f"Promedio de F1-Score: {np.mean(f1_scores):.4f}")

# Calcular estadísticas de los 35 experimentos
print("\nEstadísticas de los experimentos:")
print(f"Exactitud - Media: {np.mean(accuracies):.4f}, Desv. Estándar: {np.std(accuracies):.4f}, Mínimo: {np.min(accuracies):.4f}, Máximo: {np.max(accuracies):.4f}")
print(f"Precision - Media: {np.mean(precisions):.4f}, Desv. Estándar: {np.std(precisions):.4f}, Mínimo: {np.min(precisions):.4f}, Máximo: {np.max(precisions):.4f}")
print(f"Recall - Media: {np.mean(recalls):.4f}, Desv. Estándar: {np.std(recalls):.4f}, Mínimo: {np.min(recalls):.4f}, Máximo: {np.max(recalls):.4f}")
print(f"F1-Score - Media: {np.mean(f1_scores):.4f}, Desv. Estándar: {np.std(f1_scores):.4f}, Mínimo: {np.min(f1_scores):.4f}, Máximo: {np.max(f1_scores):.4f}")