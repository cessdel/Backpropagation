import numpy as np
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Función sigmoide y su derivada
def sigmoid(x):
    x = np.clip(x, -500, 500)  # Evitar overflow
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(sig_x):
    return sig_x * (1 - sig_x)

# Cargar dataset Iris
iris = datasets.load_iris()
X = iris.data  # 4 características
y = iris.target.reshape(-1, 1)  # Etiquetas en formato columna

# Normalizar datos a rango [0,1]
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Convertir etiquetas a one-hot encoding
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y)

# Inicializar pesos
def initialize_weights():
    Vij = np.random.uniform(-1, 1, (5, 7))  # 4 entradas + bias -> 3 neuronas ocultas
    Wjk = np.random.uniform(-1, 1, (8, 3))  # 3 ocultas + bias -> 3 salidas
    return Vij, Wjk

# Parámetros de entrenamiento
alpha = 0.3
epochs = 10000
epsilon = 1e-6  # Condición de paro basada en cambio mínimo de error
experiments = 35

# Almacenar resultados
accuracies, precisions, recalls, f1_scores, confusion_matrices = [], [], [], [], []
errors_list, accs_list, test_accuracies = [], [], []


for experiment in range(experiments):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    Vij, Wjk = initialize_weights()
    prev_error = float('inf')
    
    errors, accs = [], []  # Para almacenar error y accuracy en cada epoch

    # Entrenamiento
    for epoch in range(epochs):
        total_error = 0
        correct_predictions = 0
        
        for i in range(X_train.shape[0]):
            X_sample = X_train[i]
            T_sample = y_train[i]

            # Feedforward
            Zinj = np.dot(np.append(1, X_sample), Vij)
            Zj = sigmoid(Zinj)
            Yink = np.dot(np.append(1, Zj), Wjk)
            Yk = sigmoid(Yink)

            # Calcular error cuadrático medio
            total_error += np.sum((T_sample - Yk) ** 2)
            
            # Predicción correcta
            if np.argmax(Yk) == np.argmax(T_sample):
                correct_predictions += 1

            # Backpropagation
            dk = (T_sample - Yk) * sigmoid_derivative(Yk)
            delta_Wjk = alpha * np.outer(np.append(1, Zj), dk)
            dj = np.dot(Wjk[1:], dk) * sigmoid_derivative(Zj)
            delta_Vij = alpha * np.outer(np.append(1, X_sample), dj)

            Vij += delta_Vij
            Wjk += delta_Wjk

        # Guardar error y accuracy
        total_error /= X_train.shape[0]
        errors.append(total_error)
        accs.append((correct_predictions / X_train.shape[0]) * 100)
        
        if abs(prev_error - total_error) < epsilon:
            break
        prev_error = total_error
    
    errors_list.append(errors)
    accs_list.append(accs)
    
    # Evaluación
    y_pred, y_true = [], []
    for i in range(X_test.shape[0]): # Iteramos sobre todas las muestras de prueba
        X_sample = X_test[i] # Extraemos una muestra del conjunto de prueba
        T_sample = y_test[i] # Extraemos su etiqueta real (one-hot encoding)
        
        # Cálculo de la salida de la capa oculta
        Zinj = np.dot(np.append(1, X_sample), Vij) # Producto punto entre entrada y pesos
        Zj = sigmoid(Zinj)  # Aplicación de la función sigmoide
        Yink = np.dot(np.append(1, Zj), Wjk) # Producto punto con la segunda capa de pesos
        Yk = sigmoid(Yink) # Aplicación de la función sigmoide para obtener probabilidades
        
        # Guardamos la clase predicha (índice de la mayor probabilidad) y la clase real
        y_pred.append(np.argmax(Yk)) # Índice de la mayor activación en la capa de salida
        y_true.append(np.argmax(T_sample)) # Índice del 1 en el vector one-hot encoding

    # Calcular métricas
    acc = accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred, average='macro') * 100
    rec = recall_score(y_true, y_pred, average='macro') * 100
    f1 = f1_score(y_true, y_pred, average='macro') * 100
    conf_matrix = confusion_matrix(y_true, y_pred)

    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)
    f1_scores.append(f1)
    confusion_matrices.append(conf_matrix)
    test_accuracies.append(acc)

# Graficar error y accuracy promedio
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
for errors in errors_list:
    plt.plot(errors, alpha=0.3)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Error durante el entrenamiento (Promedio de experimentos)')

plt.subplot(1, 2, 2)
for accs in accs_list:
    plt.plot(accs, alpha=0.3)
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy durante el entrenamiento (Promedio de experimentos)')
plt.show()

# Graficar accuracy en el conjunto de prueba
epochs_range = range(1, experiments + 1)
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, test_accuracies, marker='o', linestyle='-', color='blue', label='Test Accuracy')
plt.xlabel('Experimentos')
plt.ylabel('Exactitud (%)')
plt.title('Evolución de la Exactitud en el Conjunto de Prueba')
plt.legend()
plt.grid()
plt.show()

# Resultados finales
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.hist(accuracies, bins=10, color='blue', alpha=0.7)
plt.xlabel('Accuracy (%)')
plt.ylabel('Frecuencia')
plt.title('Distribución de Accuracy')

plt.subplot(2, 2, 2)
plt.hist(precisions, bins=10, color='green', alpha=0.7)
plt.xlabel('Precision (%)')
plt.ylabel('Frecuencia')
plt.title('Distribución de Precision')

plt.subplot(2, 2, 3)
plt.hist(recalls, bins=10, color='red', alpha=0.7)
plt.xlabel('Recall (%)')
plt.ylabel('Frecuencia')
plt.title('Distribución de Recall')

plt.subplot(2, 2, 4)
plt.hist(f1_scores, bins=10, color='purple', alpha=0.7)
plt.xlabel('F1-Score (%)')
plt.ylabel('Frecuencia')
plt.title('Distribución de F1-Score')

plt.tight_layout()
plt.show()

# Mostrar última matriz de confusión
plt.figure(figsize=(6, 6))
sns.heatmap(confusion_matrices[-1], annot=True, fmt='d', cmap='Blues')
plt.title("Matriz de Confusión (último experimento)")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()
