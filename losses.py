import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def categorical_crossentropy(y_pred, y_true):
    probs = y_pred[np.arange(len(y_pred)), y_true]
    entropy = - probs + np.log(np.sum(np.exp(y_pred), axis=-1))
    return entropy

def grad_categorical_crossentropy(y_pred, y_true):
    probs = np.zeros_like(y_pred)
    probs[np.arange(len(y_pred)), y_true] = 1

    grad_entropy = np.exp(y_pred) / np.exp(y_pred).sum(axis=-1, keepdims=True)

    return (-probs + grad_entropy) / y_pred.shape[0]

def binary_crossentropy(y_pred, y_true):
    z = sigmoid(y_pred)
    loss = (y * np.log(1e-15 + z)) + ((1-y) * np.log(1-(1e-15 + z)))
    return -loss

def grad_binary_crossentropy(y_pred, y_true):
    z = sigmoid(y_pred)
    return (z - y_true)

def mse(y_pred, y_true):
    return np.square((y_true - y_pred))

def grad_mse(y_true, y_pred):
    return -(2 * (y_true - y_pred))