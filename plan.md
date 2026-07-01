# Project Plan: MLP on Breast Cancer Data

## Phase 1 — Learn the concepts — 2–3h
- [x] Watch 3Blue1Brown's neural network series, especially the backprop/calculus episodes
- [x] Read CS231n backpropagation notes (cs231n.github.io/optimization-2)
- [x] Be able to answer without notes:
  - What does a forward pass compute, step by step?
  - What is a gradient, and why do we need it?
  - What does the chain rule have to do with backpropagation?
  - Why does gradient descent move against the gradient?

## Phase 2 — Derive the math on paper — 1.5–2h
- [x] Write out the forward pass equations by hand: input → linear → ReLU → linear → sigmoid → loss
- [x] Derive all four gradients (∂L/∂W2, ∂L/∂b2, ∂L/∂W1, ∂L/∂b1) using chain rule, every step shown
- [x] Check derivation against CS231n notes

## Phase 3 — Build the data loader — 1–1.5h
- [x] Load breast cancer dataset, inspect shapes and actual values
- [x] Write normalization function (subtract mean, divide by std, per feature)
- [x] Write shuffle and train/test split function
- [x] Write mini-batch generator using yield
- [x] Test in isolation — print shapes, confirm batches look correct

## Phase 4 — Build the model — 2–2.5h
- [x] Write parameter initialization (small random weights, zero biases)
- [x] Write forward pass function, returning intermediates for backprop
- [x] Write binary cross-entropy loss function with log(0) clipping
- [x] Write backward pass implementing hand-derived gradients
- [x] Write gradient descent parameter update

## Phase 5 — Train and debug — 1.5–2h
- [x] Write training loop: epochs → batches → forward → loss → backward → update
- [x] Write evaluation function (threshold at 0.5, compute accuracy)
- [x] Debug loss not decreasing — found broadcasting bug in loss function (y shape (32,) vs a2 shape (32,1))
- [x] Confirmed working end to end, read back through every line

## Phase 6 — Kernel methods comparison — 1h
- [x] Run sklearn.svm.SVC(kernel='rbf') on same data, compare accuracy
- [x] Read about RBF kernel trick and be able to explain without notes
