# Phase 1. learn the concept
## 3B1B deep learning series
learning -> find the right weights and biases

cost function is a layer of complexity on top of neural network function
- neural network function: input -> 784 numbers; output -> 10 numbers; parameters -> 13002 weights & biases
- cost function: input -> 13002 weights & biases; output -> one number (cost); parameters -> many many training examples

backpropogation: an algorithm for determining how a single training example would like to nudge the weights and biases, specifically, what relative proportions to those changes cause the most rapid decrease to the cost. Stochastic gradient descent is used to is used to speed up computation. 

calculus for backpropagation
$$\frac{\partial C_0}{\partial w^{(L)}} = \underbrace{\frac{\partial z^{(L)}}{\partial w^{(L)}}}_{\text{nudge to } w^{(L)} \to z^{(L)}} \cdot \underbrace{\frac{\partial a^{(L)}}{\partial z^{(L)}}}_{\text{nudge to } z^{(L)} \to a^{(L)}} \cdot \underbrace{\frac{\partial C_0}{\partial a^{(L)}}}_{\text{nudge to } a^{(L)} \to C_0}$$

$$z^{(L)} = w^{(L)} a^{(L-1)} + b^{(L)} \quad\longrightarrow\quad \frac{\partial z^{(L)}}{\partial w^{(L)}} = a^{(L-1)}$$

$$a^{(L)} = \sigma\left(z^{(L)}\right) \quad\longrightarrow\quad \frac{\partial a^{(L)}}{\partial z^{(L)}} = \sigma'\left(z^{(L)}\right)$$

$$C_0 = \left(a^{(L)} - y\right)^2 \quad\longrightarrow\quad \frac{\partial C_0}{\partial a^{(L)}} = 2\left(a^{(L)} - y\right)$$
