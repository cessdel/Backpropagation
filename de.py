import numpy as np
from sklearn import datasets
from funciones import sphere, rastrigin, rosenbrock, ackley
import matplotlib.pyplot as plt
import time
import pandas as pd
from scipy import stats

# Parámetros de la evolución diferencial
N = 30  # Tamaño de la población
dim = 10  # Dimensión de cada individuo
X_min = -1 * np.ones(dim)  # Rango mínimo
X_max = 1 * np.ones(dim)  # Rango máximo
# X_min = -5 * np.ones(dim) #Para la funcion Rosenbrock
# X_max = 10 * np.ones(dim) #Para la función Rosenbrock

# Parámetros de evolución diferencial
F = 0.9  # Factor de escala
CR = 0.8  # Tasa de cruza
generaciones = 1000*dim  # Número de generaciones

# Número de repeticiones del experimento
num_experimentos = 35
mejores_por_experimento = [] #mejor fitness de cada experimento
tiempos_de_ejecucion = [] # Almacena los tiempos de ejecución de cada experimento

# Almacena todos los mejores fitness para todas las generaciones de todos los experimentos
todos_los_experimentos = [] 
vectores_optimos = []

# === Bucle para repetir el experimento 35 veces ===
for exp in range(num_experimentos):
    #print(f"\n====================\nExperimento {exp+1}/{num_experimentos}")
    inicio = time.time()

    # Inicializar la población
    population = np.random.uniform(X_min, X_max, (N, dim))

    # Calcular fitness inicial
    fitness = np.array([rastrigin(individuo) for individuo in population])

    mejores_por_gen = []

    for gen in range(generaciones):
        nueva_poblacion = []  # Nueva población en cada generación
        for i in range(N):
            # === MUTACIÓN ===
            otros = np.delete(np.arange(N), i) # Eliminar el índice del individuo actual
            r1, r2, r3 = np.random.choice(otros, 3, replace=False)
            X_m = population[r1] + F * (population[r2] - population[r3])

            # === CRUCE ===
            jrand = np.random.randint(dim)  
            trial = np.copy(population[i])
            for j in range(dim):
                if np.random.rand() < CR or j == jrand:
                    trial[j] = X_m[j]

            # === EVALUACIÓN ===
            f_trial = rastrigin(trial)

            # === SELECCIÓN ===
            if f_trial < fitness[i]:
                nueva_poblacion.append(trial)
                fitness[i] = f_trial
            else:
                nueva_poblacion.append(population[i])

        # Actualizar población
        population = np.array(nueva_poblacion)
        mejor_fitness = np.min(fitness)
        indice_mejor = np.argmin(fitness)
        mejor_vector = population[indice_mejor]
        mejores_por_gen.append(mejor_fitness)

    # Almacenar el mejor fitness de este experimento
    mejores_por_experimento.append(mejor_fitness)
    todos_los_experimentos.append(mejores_por_gen)
    vectores_optimos.append(mejor_vector)


    fin = time.time()
    tiempos_de_ejecucion.append(fin - inicio)
    #print(f"Mejor fitness al final del experimento {exp+1}: {np.format_float_scientific(mejor_fitness, precision=4)}")

# === Análisis de resultados ===
mejores_por_experimento = np.array(mejores_por_experimento)
tiempo_promedio = np.mean(tiempos_de_ejecucion)

# Estadísticas de los mejores fitness finales
estadisticas = {
    "Media": np.mean(mejores_por_experimento),
    "Desviación estándar": np.std(mejores_por_experimento),
    "Moda": stats.mode(mejores_por_experimento, keepdims=True)[0][0],
    "Mediana": np.median(mejores_por_experimento),
    "Máximo": np.max(mejores_por_experimento),
    "Mínimo": np.min(mejores_por_experimento),
    "Vector óptimo": vectores_optimos[np.argmin(mejores_por_experimento)],
    "Tiempo promedio (s)": tiempo_promedio
}

# Crear tabla de resultados
df_estadisticas = pd.DataFrame([estadisticas])

print("\n=== Estadísticas de los mejores fitness finales ===")
print(df_estadisticas)
for i, vec in enumerate(vectores_optimos):
    print(f"Experimento {i+1}: Vector óptimo = {vec}")

# === Graficar la evolución del fitness para todos los experimentos ===
plt.figure(figsize=(10, 6))
for i in range(num_experimentos):
    plt.plot(todos_los_experimentos[i], alpha=0.5, label=f"Exp {i+1}" if i < 3 else "")

plt.xlabel('Generación')
plt.ylabel('Mejor Fitness')
plt.title('Evolución del Fitness para 35 Experimentos')
plt.grid(True)
plt.legend(loc="upper right", ncol=2, fontsize="small")
plt.tight_layout()
plt.show()
#Hisotgrama
plt.figure()
plt.hist(mejores_por_experimento, bins=10, edgecolor='black')
plt.title('Distribución del Mejor Fitness Final')
plt.xlabel('Fitness')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.tight_layout()
plt.show()

# === Guardar resultados en un archivo CSV ===
df_estadisticas.to_csv('estadisticas_experimentos.csv', index=False)
