import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def initialize_centroids(X, K):
    """
    Inicializa aleatoriamente K centroides a partir del conjunto de datos X.
    
    Parámetros:
    X: np.array de forma (N, D) -> Datos con N puntos y D características.
    K: int -> Número de clústeres.

    Retorna:
    centroids: np.array de forma (K, D) -> K centroides seleccionados aleatoriamente.
    """
    indices = np.random.choice(len(X), K, replace=False)
    return X[indices]

def assign_clusters(X, centroids):
    """
    Asigna cada punto de X al centroide más cercano.
    
    Parámetros:
    X: np.array de forma (N, D) -> Datos con N puntos y D características.
    centroids: np.array de forma (K, D) -> K centroides.

    Retorna:
    labels: np.array de forma (N,) -> Índice del centroide más cercano para cada punto.
    """
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(X, labels, K):
    """
    Recalcula los centroides como la media de los puntos asignados a cada clúster.
    
    Parámetros:
    X: np.array de forma (N, D) -> Datos con N puntos y D características.
    labels: np.array de forma (N,) -> Índice del centroide más cercano para cada punto.
    K: int -> Número de clústeres.

    Retorna:
    centroids: np.array de forma (K, D) -> Nuevos centroides."""
    
    return np.array([X[labels == k].mean(axis=0) for k in range(K)])

def kmeans(X, K, tol=1e-4, max_iter=100):
    """
    Ejecuta el algoritmo K-Means de forma manual.
    
    Parámetros:
    X: np.array de forma (N, D) -> Datos con N puntos y D características.
    K: int -> Número de clústeres.
    tol: float -> Umbral de convergencia.
    max_iter: int -> Número máximo de iteraciones.

    Retorna:
    centroids: np.array de forma (K, D) -> Centroides finales.
    labels: np.array de forma (N,) -> Índice del centroide más cercano para cada punto.
    """
    centroids = initialize_centroids(X, K)
    initial_centroids = centroids.copy()
    
    for i in range(max_iter):
        old_centroids = centroids.copy()
        labels = assign_clusters(X, centroids)
        centroids = update_centroids(X, labels, K)
        
        if np.linalg.norm(centroids - old_centroids) < tol:
            break
    
    return initial_centroids, centroids, labels

# Cargamos el dataset Iris
iris = datasets.load_iris()
X=iris.data
Y = iris.target.reshape(-1, 1)  # Convertimos en columna
# Convertir etiquetas a One-Hot Encoding
encoder = OneHotEncoder(sparse_output=False)
Y_one_hot = encoder.fit_transform(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y_one_hot, test_size=0.2, random_state=42)

# Ejecutamos K-Means
K = 3
initial_centroids, final_centroids, labels = kmeans(X_train, K)

# Visualización de los centroides iniciales y finales en una sola figura
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

print(f"Final centroids: {final_centroids}")
# Gráfico de centroides iniciales
axes[0].scatter(X_train[:, 0], X_train[:, 1], c='gray', alpha=0.5, label='Puntos de datos')
axes[0].scatter(initial_centroids[:, 0], initial_centroids[:, 1], c='red', marker='x', s=100, label='Centroides Iniciales')
axes[0].set_xlabel('Sepal length')
axes[0].set_ylabel('Sepal width')
axes[0].set_title('Centroides Iniciales')
axes[0].legend()

# Gráfico de centroides finales
axes[1].scatter(X_train[:, 0], X_train[:, 1], c=labels, cmap='viridis', alpha=0.6, label='Puntos de datos')
axes[1].scatter(final_centroids[:, 0], final_centroids[:, 1], c='red', marker='x', s=100, label='Centroides Finales')
axes[1].set_xlabel('Sepal length')
axes[1].set_ylabel('Sepal width')
axes[1].set_title('Centroides Finales')
axes[1].legend()

plt.suptitle('Comparación de Centroides Iniciales y Finales en K-Means')
plt.show()

#------------------------------------------ base radial ------------------------------------------
from sklearn.preprocessing import OneHotEncoder

def gaussian_function(X, centers, sigma):
    """Calcula la activación de la capa intermedia usando funciones de base radial."""
    return np.exp(-np.linalg.norm(X[:, np.newaxis] - centers, axis=2) ** 2 / (2 * sigma ** 2))

def initialize_weights(input_size, output_size):
    """Inicializa los pesos de manera uniforme en el rango [-1,1]."""
    return np.random.uniform(-1, 1, size=(input_size, output_size)), np.random.uniform(-1, 1, size=(1, output_size))

def gaussian_function(X, centers, d):
    """Calcula la activación de la capa intermedia usando funciones de base radial (Gaussiana)."""
    r = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)  # Distancia euclidiana
    return np.exp(- (r ** 2) / (2 * d ** 2))

def initialize_weights(input_size, output_size):
    """Inicializa los pesos de manera uniforme en el rango [-1,1]."""
    return np.random.uniform(-1, 1, size=(input_size, output_size)), np.random.uniform(-1, 1, size=(1, output_size))

def train_rbf(X_train, y_train, centers, d, alpha, epochs):
    """Entrena la Red Neuronal de Base Radial usando la regla delta."""
    M = centers.shape[0]  # Número de neuronas en la capa intermedia (igual a las clases)
    K = y_train.shape[1]  # Número de neuronas en la capa de salida
    
    # Inicialización de pesos
    W, W0 = initialize_weights(M, K)
    
    print(f"W: {W}")
    print(f"W0: {W0}")
    # Entrenamiento
    for epoch in range(epochs):
        for X, t in zip(X_train, y_train):
            
            # Capa intermedia (activaciones RBF)
            phi = gaussian_function(X.reshape(1, -1), centers, d)
            
            # Capa de salida (activación lineal)
            Y = phi @ W + W0
            
            # Error
            error = t - Y
            
            # Regla delta para actualizar pesos
            delta_W = alpha * error * phi.T
            delta_W0 = alpha * error
            
            W += delta_W
            W0 += delta_W0
            
            
    
    return W, W0

def predict_rbf(X, centers, d, W, W0):
    """Realiza la predicción con la RBF entrenada."""
    phi = gaussian_function(X, centers, d)
    Y = phi @ W + W0
    return np.argmax(Y, axis=1)  # Devuelve la clase con mayor activación






# Definir parámetros
d = 1.0  # Parámetro de la Gaussiana (equivalente a di en la ecuación)
alpha = 0.01  # Tasa de aprendizaje
epochs = 1  # Número de épocas

# Usar los centroides de K-Means como centros RBF
centers = final_centroids

# Entrenar la Red Neuronal RBF
W, W0 = train_rbf(X_train, y_train, centers, d, alpha, epochs)

# Realizar predicciones
y_pred = predict_rbf(X_test, centers, d, W, W0)
y_test_labels = np.argmax(y_test, axis=1)

# Calcular precisión
accuracy = np.mean(y_pred == y_test_labels)
print(f'Precisión de la Red Neuronal de Base Radial: {accuracy * 100:.2f}%')
