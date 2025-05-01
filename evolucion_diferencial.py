import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from perceptron_feedforward import fitness_function, forward_pass_DE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix   

# ==========================
# Preparar datos (Iris)
# ==========================
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y)


# ==========================

num_experimentos = 35

# ==========================
# Almacenamiento de resultados
# ==========================
mejores_por_experimento = []
tiempos_de_ejecucion = []
vectores_optimos = []
accuracies = []
precisions = []
recalls = []
f1_scores = []
todos_los_experimentos = []
train_accuracies = []
test_accuracies = []


import time
for exp in range(num_experimentos):
    # Parámetros DE
    # ==========================
    N = 20             # Tamaño de la población
    dim = 59           # Número total de pesos (Vij + Wjk)
    X_min = -5 * np.ones(dim)
    X_max = 5 * np.ones(dim)

    F = .15
    CR = 0.8
    generaciones = 5000
    inicio = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)


    # Inicializar población
    population = np.random.uniform(X_min, X_max, (N, dim))
    fitness = np.array([fitness_function(ind, X_train, y_train) for ind in population])

    mejores_por_gen = []

    for gen in range(generaciones):
        nueva_poblacion = []
        for i in range(N):
            # Seleccionamos el mejor individuo
            best_index = np.argmin(fitness)
            x_best = population[best_index]

            # Seleccionamos r1 y r2, distintos de i
            otros = np.delete(np.arange(N), i)
            r1, r2 = np.random.choice(otros, 2, replace=False)

            # Mutación current-to-best/1
            mutant = population[i] + F * (x_best - population[i]) + F * (population[r1] - population[r2])
            jrand = np.random.randint(dim)
            trial = np.where((np.random.rand(dim) < CR) | (np.arange(dim) == jrand), mutant, population[i])

            f_trial = fitness_function(trial, X_train, y_train)

            if f_trial < fitness[i]:
                nueva_poblacion.append(trial)
                fitness[i] = f_trial
            else:
                nueva_poblacion.append(population[i])

        population = np.array(nueva_poblacion)
        mejor_fitness = np.min(fitness)
        mejores_por_gen.append(mejor_fitness)

    fin = time.time()

    mejores_por_experimento.append(mejor_fitness)
    tiempos_de_ejecucion.append(fin - inicio)
    vectores_optimos.append(population[np.argmin(fitness)])
    todos_los_experimentos.append(mejores_por_gen)
    
    
    # === Evaluación final con el mejor vector de pesos ===
    best_weights = vectores_optimos[np.argmin(mejores_por_experimento)]
    
    # Evaluación en entrenamiento (train)
    y_train_pred = forward_pass_DE(X_train, best_weights)
    y_train_pred_labels = np.argmax(y_train_pred, axis=1)
    y_train_true_labels = np.argmax(y_train, axis=1)

    train_accuracy = accuracy_score(y_train_true_labels, y_train_pred_labels)

    # Evaluación en prueba (test)
    y_test_pred = forward_pass_DE(X_test, best_weights)
    y_test_pred_labels = np.argmax(y_test_pred, axis=1)
    y_test_true_labels = np.argmax(y_test, axis=1)

    test_accuracy = accuracy_score(y_test_true_labels, y_test_pred_labels)
    test_precision = precision_score(y_test_true_labels, y_test_pred_labels, average='macro')
    test_recall = recall_score(y_test_true_labels, y_test_pred_labels, average='macro')
    test_f1 = f1_score(y_test_true_labels, y_test_pred_labels, average='macro')

    # 🔵 Guardar métricas
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)
    accuracies.append(test_accuracy)
    precisions.append(test_precision)
    recalls.append(test_recall)
    f1_scores.append(test_f1)


# ==========================
# Estadísticas
# ==========================
estadisticas = {
    "Media del Fitness": np.mean(mejores_por_experimento),
    "Desviación estándar": np.std(mejores_por_experimento),
    "Moda": stats.mode(mejores_por_experimento, keepdims=True)[0][0],
    "Mediana": np.median(mejores_por_experimento),
    "Máximo": np.max(mejores_por_experimento),
    "Mínimo": np.min(mejores_por_experimento),
    "Vector óptimo global": vectores_optimos[np.argmin(mejores_por_experimento)],
    "Tiempo promedio (s)": np.mean(tiempos_de_ejecucion)
}
# === 🟢 Crear DataFrame de métricas ===
df_metrics = pd.DataFrame({
    'Accuracy': accuracies,
    'Accuracy min': [np.min(accuracies)] * num_experimentos,
    'Accuracy mean': [np.mean(accuracies)] * num_experimentos,
    'Accuracy std': [np.std(accuracies)] * num_experimentos,
    'Accuracy median': [np.median(accuracies)] * num_experimentos,
    'Accuracy max': [np.max(accuracies)] * num_experimentos,
    'Precision': precisions,
    'Recall': recalls,
    'F1-Score': f1_scores
})

# === 🟠 Guardar resultados en CSV ===
df_metrics.to_csv('metricas_experimentos_de.csv', index=False)

# === 🟡 Mostrar estadísticas descriptivas de las métricas ===
print("\n=== Estadísticas Generales de las Métricas ===")
print(df_metrics.describe())

df_estadisticas = pd.DataFrame([estadisticas])
print("\n=== Estadísticas de los mejores fitness finales ===")
print(df_estadisticas)

# ==========================
# Gráficas
# ==========================
plt.figure(figsize=(10, 6))
for i in range(num_experimentos):
    plt.plot(todos_los_experimentos[i], alpha=0.5, label=f"Exp {i+1}" if i < 3 else "")

# 🔥 Líneas horizontales en 0.01 y 0.05:
plt.axhline(y=0.01, color='red', linestyle='--', linewidth=1, label='0.01 (Referencia)')
plt.axhline(y=0.05, color='green', linestyle='--', linewidth=1, label='0.05 (Referencia)')

plt.xlabel('Generación')
plt.ylabel('Mejor Fitness')
plt.title('Evolución del Fitness por Experimento (Iris)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# 🟢 Gráfico de Accuracy de Train y Test a lo largo de los experimentos
plt.figure(figsize=(10, 6))
experiments_range = range(1, len(train_accuracies) + 1)

plt.plot(experiments_range, train_accuracies, label='Train Accuracy', marker='o', linestyle='-', color='blue')
plt.plot(experiments_range, test_accuracies, label='Test Accuracy', marker='s', linestyle='-', color='orange')

plt.title('Accuracy en Train y Test a lo largo de los Experimentos')
plt.xlabel('Número de Experimento')
plt.ylabel('Accuracy')
plt.ylim(0.8, 1.0)  # 🔥 Aquí cambias el límite del eje Y
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('accuracy_train_test_experimentos.png')  # Opcional: guardar la imagen
plt.show()


# ===== Evaluación FINAL sobre TODO el dataset Iris =====
# Volvemos a cargar los datos completos (sin split)
iris = load_iris()
X_full = iris.data
y_full = iris.target.reshape(-1, 1)

encoder = OneHotEncoder(sparse_output=False)
y_full = encoder.fit_transform(y_full)

# Tomamos el mejor vector óptimo global
best_weights = vectores_optimos[np.argmin(mejores_por_experimento)]

# Evaluamos sobre TODO el dataset
y_full_pred = forward_pass_DE(X_full, best_weights)
y_full_pred_labels = np.argmax(y_full_pred, axis=1)
y_full_true_labels = np.argmax(y_full, axis=1)

# Métricas sobre todo el conjunto
accuracy_full = accuracy_score(y_full_true_labels, y_full_pred_labels)
precision_full = precision_score(y_full_true_labels, y_full_pred_labels, average='macro')
recall_full = recall_score(y_full_true_labels, y_full_pred_labels, average='macro')
f1_full = f1_score(y_full_true_labels, y_full_pred_labels, average='macro')

print("\n=== Evaluación FINAL sobre TODO el dataset Iris ===")
print(f"Accuracy : {accuracy_full:.4f}")
print(f"Precision: {precision_full:.4f}")
print(f"Recall   : {recall_full:.4f}")
print(f"F1-Score : {f1_full:.4f}")

# ===== Matriz de confusión sobre TODO el dataset =====
conf_matrix_full = confusion_matrix(y_full_true_labels, y_full_pred_labels)
plt.figure(figsize=(6, 6))
import seaborn as sns
sns.heatmap(conf_matrix_full, annot=True, fmt='d', cmap='Blues')
plt.title("Matriz de Confusión (Todo el Dataset Iris)")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()

# Guardar resultados
df_estadisticas.to_csv('estadisticas_experimentos_iris_de.csv', index=False)
np.save('vectores_optimos_iris_de.npy', np.array(vectores_optimos))
