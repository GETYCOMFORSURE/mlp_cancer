# Phase 1. learn the concept
## 3B1B deep learning series
learning -> find the right weights and biases

cost function is a layer of complexity on top of neural network function
- neural network function: input -> 784 numbers; output -> 10 numbers; parameters -> 13002 weights & biases
- cost function: input -> 13002 weights & biases; output -> one number (cost); parameters -> many many training examples

backpropogation: an algorithm for determining how a single training example would like to nudge the weights and biases, specifically, what relative proportions to those changes cause the most rapid decrease to the cost. Stochastic gradient descent is used to is used to speed up computation. 
