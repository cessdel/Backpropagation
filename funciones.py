import numpy as np

def sphere(x, debug=False):
    cuadrados = x ** 2
    suma = np.sum(cuadrados)

    if debug:
        print(f"\nEvaluando individuo:")
        print(f"Vector: {x}")
        print(f"Componentes al cuadrado: {cuadrados}")

    return suma

def sigmoid(x):
    x = np.clip(x, -500, 500)  # Limita los valores extremos para evitar overflow
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)

def rastrigin(x):
    """Función Rastrigin: muchas oscilaciones, mínimo global en el origen"""
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def rosenbrock(x):
    """Función Rosenbrock: mínimo en el vector [1, 1, ..., 1]"""
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def ackley(x):
    """Función Ackley: mínimo en el origen, superficie compleja"""
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + a + np.exp(1)

def mse(y_true, y_pred, debug=False):
    """Error Cuadrático Medio (MSE)"""
    error = np.mean((y_true - y_pred) ** 2)
    
    if debug:
        print(f"\nCalculando MSE:")
        print(f"Valores reales: {y_true}")
        print(f"Valores predichos: {y_pred}")
        print(f"Error cuadrático medio: {error}")

    return error
