import numpy as np

def he_init(next_layer,prev_layer):
  w = []
  for i in range(next_layer):
    std = np.sqrt(2.0/prev_layer)
    row = randn(prev_layer) * std
    w.append(row)
  return np.array(w)

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1/(1 + np.exp((-x)))

def sigmoid_d(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(x, 0)

def relu_d(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x