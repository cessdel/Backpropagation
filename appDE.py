import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from funciones import sphere, rastrigin, rosenbrock, ackley

# === CONFIGURACIÓN DE LA APP ===
st.set_page_config(page_title="Evolución Diferencial (Funciones)", layout="centered")
st.title("Comparativa de Evolución Diferencial con Distintas Funciones Objetivo")

# === PARÁMETROS DESDE SIDEBAR ===
st.sidebar.header("Parámetros de Evolución Diferencial")
N = st.sidebar.slider("Tamaño de la población (N)", 10, 100, 30)
dim = st.sidebar.slider("Dimensión del individuo", 2, 100, 10)
generaciones = st.sidebar.slider("Número de generaciones", 10, 10000, 1000)
F = st.sidebar.slider("Factor de mutación (F)", 0.1, 1.0, 0.5)
CR = st.sidebar.slider("Tasa de cruce (CR)", 0.0, 1.0, 0.8)

# === FUNCIONES OBJETIVO ===
funciones = {
    "Sphere": sphere,
    "Rastrigin": rastrigin,
    "Rosenbrock": rosenbrock,
    "Ackley": ackley
}

# === INICIALIZACIÓN ===
X_min = -1 * np.ones(dim)
X_max = 1 * np.ones(dim)

# === RESULTADOS ===
resultados = {}
mejores_individuos = {}
with st.spinner("🔄 Ejecutando evolución diferencial, por favor espera..."):

    for nombre, func_obj in funciones.items():
        population = np.random.uniform(X_min, X_max, (N, dim))
        fitness = np.array([func_obj(ind) for ind in population])
        mejores_por_gen = []

        for gen in range(generaciones):
            nueva_poblacion = []
            for i in range(N):
                indices = list(range(N))
                indices.remove(i)
                r1, r2, r3 = np.random.choice(indices, 3, replace=False)
                X_m = population[r1] + F * (population[r2] - population[r3])

                jrand = np.random.randint(dim)
                trial = np.copy(population[i])
                for j in range(dim):
                    if np.random.rand() < CR or j == jrand:
                        trial[j] = X_m[j]

                f_trial = func_obj(trial)
                if f_trial < fitness[i]:
                    nueva_poblacion.append(trial)
                    fitness[i] = f_trial
                else:
                    nueva_poblacion.append(population[i])

            population = np.array(nueva_poblacion)
            mejores_por_gen.append(np.min(fitness))

        resultados[nombre] = mejores_por_gen
        mejores_individuos[nombre] = (population[np.argmin(fitness)], np.min(fitness))

# === GRAFICAR EN GRID 2x2 ===
st.subheader("📊 Evolución del fitness por función objetivo")
fig = make_subplots(rows=2, cols=2, subplot_titles=list(funciones.keys()))

posiciones = [(1, 1), (1, 2), (2, 1), (2, 2)]
for (nombre, valores), (r, c) in zip(resultados.items(), posiciones):
    fig.add_trace(go.Scatter(
        y=valores,
        mode="lines",
        name=nombre,
        showlegend=False
    ), row=r, col=c)

    fig.update_xaxes(title_text="Generación", row=r, col=c)
    fig.update_yaxes(title_text="Fitness", row=r, col=c)

fig.update_layout(height=600, width=800, title_text="Evolución del fitness por función", template="plotly_white")
st.plotly_chart(fig)

# === MOSTRAR MEJORES INDIVIDUOS EN TABLA ===
st.subheader("🏆 Mejores soluciones encontradas")
data = {
    "Función": [],
    "Fitness": [],
    "Individuo": []
}

for nombre, (individuo, fitness) in mejores_individuos.items():
    data["Función"].append(nombre)
    data["Fitness"].append(f"{fitness:.8f} ({fitness:.2e})")
    data["Individuo"].append(np.array2string(individuo, precision=6, separator=', '))

df_resultados = pd.DataFrame(data)
table_fig = go.Figure(data=[go.Table(
    header=dict(values=["Función", "Fitness", "Individuo"],
                fill_color='lightblue',
                align='left',
                font=dict(color='black', size=13)),
    cells=dict(values=[
        df_resultados["Función"],
        df_resultados["Fitness"],
        df_resultados["Individuo"]
    ],
    fill_color='white',
    align='left',
    font=dict(color='black', size=12, family="Courier New"),)
)])

table_fig.update_layout(
    width=1000,
    height=400,
    margin=dict(l=0, r=0, t=10, b=0)
)

st.plotly_chart(table_fig)