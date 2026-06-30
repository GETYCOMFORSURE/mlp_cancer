from sklearn.datasets import load_breast_cancer

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
  a = x - mean_x
  x_normalized = a / std_X
  X_normalized.append(x_normalized)


  

