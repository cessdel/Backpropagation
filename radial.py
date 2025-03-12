import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Cargar el dataset Iris
iris = datasets.load_iris()
X = iris.data  # Características
y = iris.target  # Etiquetas reales (solo para comparación)

# Dividir el conjunto de datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicialización de parámetros
K = 3  # Número de clusters
epsilon = 1e-4  # Criterio de convergencia
max_iter = 100  # Máximo de iteraciones

# Inicialización aleatoria de centroides usando solo el conjunto de entrenamiento
np.random.seed(42)
centroids = X_train[np.random.choice(X_train.shape[0], K, replace=False)]

for iteration in range(max_iter):
    # Paso 1: Asignación de cada punto al centroide más cercano
    distances = np.linalg.norm(X_train[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)

    # Paso 2: Calcular nuevos centroides como la media de los puntos asignados
    new_centroids = np.array([X_train[labels == k].mean(axis=0) for k in range(K)])

    # Paso 3: Verificar convergencia
    if np.linalg.norm(new_centroids - centroids) < epsilon:
        break

    centroids = new_centroids

# Implementación de la Red Neuronal de Base Radial (RBF)
def gaussian_rbf(x, c, d):
    """Función de base radial gaussiana."""
    return np.exp(-np.linalg.norm(x - c) ** 2 / (2 * d ** 2))

# Calcular sigma (desviación estándar de los centroides)
d_max = np.max(cdist(centroids, centroids))
sigma = d_max / np.sqrt(2 * K)

# Construcción de la matriz de activaciones de la capa intermedia para entrenamiento
G_train = np.zeros((X_train.shape[0], K))
for i in range(X_train.shape[0]):
    for j in range(K):
        G_train[i, j] = gaussian_rbf(X_train[i], centroids[j], sigma)

# Entrenamiento de la red (cálculo de pesos de salida)
G_inv = np.linalg.pinv(G_train)
W = np.dot(G_inv, y_train)  # Pesos de salida

# Construcción de la matriz de activaciones para prueba
G_test = np.zeros((X_test.shape[0], K))
for i in range(X_test.shape[0]):
    for j in range(K):
        G_test[i, j] = gaussian_rbf(X_test[i], centroids[j], sigma)

# Predicción en el conjunto de prueba
y_pred = np.dot(G_test, W)
y_pred = np.round(y_pred).astype(int)  # Redondeo al entero más cercano
y_pred = np.clip(y_pred, 0, 2)  # Asegurar valores dentro del rango de clases

# Evaluación con la métrica de precisión en el conjunto de prueba
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión de la RBF en prueba: {accuracy * 100:.2f}%')

# Visualización de la Matriz de Confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.title("Matriz de Confusión - RBF")
plt.show()

# Ordenar los datos de prueba según sus clases reales para visualizar mejor
sorted_indices = np.argsort(y_test)
G_test_sorted = G_test[sorted_indices]
y_test_sorted = y_test[sorted_indices]

# Visualización de activaciones en la capa oculta (ordenado por clase real)
plt.figure(figsize=(8, 6))
plt.imshow(G_test_sorted, aspect='auto', cmap='coolwarm')
plt.colorbar(label='Nivel de activación')
plt.xlabel("Neuronas ocultas")
plt.ylabel("Ejemplos de prueba ordenados por clase")
plt.title("Activaciones de la capa oculta - RBF (Ordenadas por clase real)")
plt.show()

# Imprimir los 30 primeros vectores de prueba y sus clases reales
print("Primeros 30 vectores de prueba y sus clases reales:")
for i in range(min(30, X_test.shape[0])):
    print(f"Vector {i}: {X_test[i]}, Clase real: {y_test[i]}, Clase predicha: {y_pred[i]}")