from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = data.data      # shape (569, 30) — the features
y = data.target    # shape (569,) — the labels (0 or 1)

print(X.shape)
print(y.shape)
print(X[0])         # one actual sample, 30 numbers
print(y[0])          # its label
