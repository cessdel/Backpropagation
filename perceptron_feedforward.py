import numpy as np
from funciones import sigmoid, mse  # Usas las funciones de tu archivo funciones.py

def unpack_weights(weights):
    """
    Divide el vector de pesos plano en las matrices Vij y Wjk.
    """
    Vij = weights[:35].reshape((5, 7))  # 5 ocultas x (4 entradas + 1 bias)
    Wjk = weights[35:].reshape((8, 3))  # 3 salidas x (7 ocultas + 1 bias)
    return Vij, Wjk

def forward_pass_DE(X, weights):
    """
    Realiza el proceso feedforward para la red dada un vector plano de pesos.
    """
    Vij, Wjk = unpack_weights(weights)
    Zinj = np.dot(np.c_[np.ones(X.shape[0]), X], Vij)  # Añadimos bias a X
    Zj = sigmoid(Zinj)
    Yink = np.dot(np.c_[np.ones(Zj.shape[0]), Zj], Wjk)  # Añadimos bias a Zj
    Yk = sigmoid(Yink)
    return Yk

def fitness_function(weights, X, y_true):
    """
    Calcula el error (fitness) usando MSE entre las salidas del perceptrón y las etiquetas reales.
    """
    y_pred = forward_pass_DE(X, weights)
    return mse(y_true, y_pred)
