import numpy as np
from data_loader import X_train, y_train, X_test, y_test

# parameter initialization
def init_params():
  W1 = np.random.randn(30, 16) * 0.01
  b1 = np.zeros(16)
  W2 = np.random.randn(16, 1)
  b2 = np.zeros(1)
  return W1, b1, W2, b2

# forward pass function
def forward(X, W1, b1, W2, b2):
  z1 = np.dot(X, W1) + b1  
  a1 = np.maximum(0, z1) 
  z2 = np.dot(a1, W2) + b2 
  a2 = 1 / (1 + np.exp(-z2)) 
  return z1, a1, z2, a2           

# loss function (binary cross-entropy)
def loss_function(y, a2):
    a2 = a2.reshape(-1)  # flatten to (32,)
    a2 = np.clip(a2, 1e-10, 1 - 1e-10)
    L = -np.mean(y * np.log(a2) + (1-y) * np.log(1-a2))
    return L

# backward pass function
def backward(X, y, a2, z1, a1, W2):
  y = y.reshape(-1, 1)
  dz2 = a2 - y # dz2 means ∂L/∂z2
  dW2 = np.dot(a1.T, dz2) / len(y)
  db2 = np.mean(dz2)
  da1 = np.dot(dz2, W2.T)
  dz1 = da1 * (z1 > 0) 
  dW1 = np.dot(np.transpose(X), dz1) / len(y)
  db1 = np.mean(dz1)
  return dW1, db1, dW2, db2

# parameter update
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    return W1, b1, W2, b2

if __name__ == "__main__": 
  W1, b1, W2, b2 = init_params()
  z1, a1, z2, a2 = forward(X_train, W1, b1, W2, b2)
  L = loss_function(y_train, a2)
  dW1, db1, dW2, db2 = backward(X_train, y_train, a2, z1, a1, W2)
  W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate = 0.01)
  print(W1, b1, W2, b2)
