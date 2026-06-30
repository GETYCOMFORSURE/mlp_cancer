import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X = data.data
y = data.target

print(X.shape)
print(y.shape)
print(X[0])         
print(y[0])          

# formula for normalization: X_normalized = (X - mean(X)) / std(X)
X_normalized = []
for x in X:
  mean_x = np.mean(X, axis=i)
  std_X = np.std(X, axis=i)
  a = x - mean_x
  x_normalized = a / std_X
  X_normalized.append(x_normalized)

# split into train / test
X_shuffle = np.random.permutation(X_normalized)
split = int(0.6 * len(X_shuffle))
X_train = X[:split] # 60% data used for train
X_test = X[split:]
