# Backpropagation derivation (hidden layer MLP)

**Notation key:**
- X: input data, shape (batch_size, 30) — 30 features per sample
- y: true labels, shape (batch_size,) — 0 or 1
- W1: weights, input → hidden, shape (30, 16)
- b1: bias, hidden layer, shape (16,)
- W2: weights, hidden → output, shape (16, 1)
- b2: bias, output layer, shape (1,)
- z1: pre-activation, hidden layer = W1·X + b1, shape (batch_size, 16)
- a1: post-activation, hidden layer = ReLU(z1), shape (batch_size, 16)
- z2: pre-activation, output layer = W2·a1 + b2, shape (batch_size, 1)
- a2: post-activation, output = σ(z2), shape (batch_size, 1) — predicted probability
- L: loss (binary cross-entropy), scalar
- σ: sigmoid function
- ReLU: rectified linear unit = max(0, z)

**Forward pass:**
- z1 = W1·X + b1
- a1 = ReLU(z1) = max(0, z1)
- z2 = W2·a1 + b2
- a2 = σ(z2) = 1 / (1 + e^(-z2))
- L = -[y·log(a2) + (1-y)·log(1-a2)]

**Sigmoid derivative:**
- σ'(z) = σ(z)·(1 - σ(z)) = a2·(1 - a2)

**Key simplification (cross-entropy + sigmoid):**
- ∂L/∂a2 = -y/a2 + (1-y)/(1-a2)
- ∂a2/∂z2 = a2·(1 - a2)
- Combined: ∂L/∂z2 = a2 - y

**Gradients (backward pass):**

∂C/∂W2 = (a2 - y) · a1

∂C/∂b2 = (a2 - y)

∂C/∂W1:
- if z1 > 0: (a2 - y) · W2 · X
- else: 0

∂C/∂b1:
- if z1 > 0: (a2 - y) · W2
- else: 0

**ReLU derivative:**
- ReLU'(z) = 1 if z > 0, else 0
