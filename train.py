from data_loader import X_train, y_train, X_test, y_test, get_batch
from model import init_params, forward, loss_function, backward, update_params


# training loop
# Write the training loop: iterate epochs, loop over batches, forward → loss → backward → update, track loss over time.
def train(X, y, epochs=100, learning_rate=0.01):
    W1, b1, W2, b2 = init_params()
    losses = []
    
    for epoch in range(epochs):
        for X_batch, y_batch in get_batch(X, y):
          z1, a1, z2, a2 = forward(X_batch, W1, b1, W2, b2)
          L = loss_function(y_batch, a2)
          dW1, db1, dW2, db2 = backward(X_batch, y_batch, a2, z1, a1, W2)
          W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        
        losses.append(L)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {L:.4f}")
    
    return W1, b1, W2, b2, losses


# evaluation function
def evaluate(X_test, y_test, W1, b1, W2, b2):
  z1, a1, z2, a2 = forward(X_test, W1, b1, W2, b2)
  predictions = (a2 > 0.5).astype(int).reshape(-1)
  accuracy = np.mean(predictions == y_test)
  return accuracy

accuracy = evaluate(X_test, y_test, W1, b1, W2, b2)
print(accuracy)
