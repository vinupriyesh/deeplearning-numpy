import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def tanh(z):
    return np.tanh(z)
def relu(z):
    return z * (z > 0)
def softmax(z):
    e = np.exp(z - np.max(z,axis=0))
    return e/e.sum(axis=0)
def leaky_relu(z, epsilon=0.1):
    return np.maximum(epsilon * z, z)
def inverse_tanh(z,a):
    return (1-np.power(a,2))
def inverse_relu(z,a):
    dz = np.array(z,copy = True)
    dz[z <= 0] = 0
    return dz
def inverse_leaky_relu(z,a, epsilon=0.1):
    gradients = 1. * (z > epsilon)
    gradients[gradients == 0] = epsilon
    return gradients
