import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# take a glimpse of the data
data = load_breast_cancer()
X = data.data
y = data.target

print(X.shape)
print(y.shape)
print(X[0])         
print(y[0])     
print(data.feature_names)

# formula for normalization: X_normalized = (X - mean(X)) / std(X)
X_normalized = []
mean_x = np.mean(X, axis=0) # axis=0 means "collapse going down the rows, for each column separately" 
std_X = np.std(X, axis=0)
for x in X:
  a = x - mean_x
  x_normalized = a / std_X
  X_normalized.append(x_normalized)
print(mean_x)
print(std_X)
print(X_normalized[0])

# split into train / test
indices = np.random.permutation(len(X_normalized)) # make sure shuffled x has corresponding y
X_shuffle = np.array(X_normalized)[indices]
y_shuffle = y[indices]
split = int(0.6 * len(X_shuffle)) # 60% data used for train
X_train = X_shuffle[:split]
X_test = X_shuffle[split:]
y_train = y_shuffle[:split]
y_test = y_shuffle[split:]

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
