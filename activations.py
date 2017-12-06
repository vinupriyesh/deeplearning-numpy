import numpy as np

def sigmoid(z):
	s = 1 / (1 + np.exp(-z))
	return s
def tanh(x):
	return np.tanh(x)
def relu(x):
	return x * (x > 0)
def leaky_relu(x, epsilon=0.1):
    return np.maximum(epsilon * x, x)
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
