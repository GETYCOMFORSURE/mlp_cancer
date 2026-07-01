# Learn the concept
## [3B1B deep learning series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
learning -> find the right weights and biases

cost function is a layer of complexity on top of neural network function
- neural network function: input -> 784 numbers; output -> 10 numbers; parameters -> 13002 weights & biases
- cost function: input -> 13002 weights & biases; output -> one number (cost); parameters -> many many training examples

backpropogation: an algorithm for determining how a single training example would like to nudge the weights and biases, specifically, what relative proportions to those changes cause the most rapid decrease to the cost. Stochastic gradient descent is used to is used to speed up computation. 

calculus for backpropagation:
<img width="1383" height="729" alt="Screenshot 2026-07-01 at 12 00 33" src="https://github.com/user-attachments/assets/56f4f65f-7e44-484a-9294-bb6d46fd4811" />

$$\frac{\partial C_0}{\partial w^{(L)}} = \underbrace{\frac{\partial z^{(L)}}{\partial w^{(L)}}}_{\text{nudge to } w^{(L)} \to z^{(L)}} \cdot \underbrace{\frac{\partial a^{(L)}}{\partial z^{(L)}}}_{\text{nudge to } z^{(L)} \to a^{(L)}} \cdot \underbrace{\frac{\partial C_0}{\partial a^{(L)}}}_{\text{nudge to } a^{(L)} \to C_0}$$

$$z^{(L)} = w^{(L)} a^{(L-1)} + b^{(L)} \quad\longrightarrow\quad \frac{\partial z^{(L)}}{\partial w^{(L)}} = a^{(L-1)}$$

$$a^{(L)} = \sigma\left(z^{(L)}\right) \quad\longrightarrow\quad \frac{\partial a^{(L)}}{\partial z^{(L)}} = \sigma'\left(z^{(L)}\right)$$

$$C_0 = \left(a^{(L)} - y\right)^2 \quad\longrightarrow\quad \frac{\partial C_0}{\partial a^{(L)}} = 2\left(a^{(L)} - y\right)$$

## [CS231n deep learning for computer vision](https://cs231n.github.io/optimization-2/)
Each gate in a circuit computes two things:
1. Its output (forward pass)
2. Its local gradient — how much its output changes per unit change in each input

During the backward pass, each gate receives the upstream gradient (how much the 
final output cares about this gate's output), then multiplies it by its own local 
gradient to get the gradient for each input. This is the chain rule.

**Example:** x=-2, y=5, z=-4, with q = x+y and f = q*z

Forward pass: q = 3, f = -12

Backward pass:
- ∂f/∂q = z = -4  (upstream gradient arriving at the add gate)
- add gate local gradient: ∂q/∂x = 1, ∂q/∂y = 1
- chain rule: ∂f/∂x = -4 * 1 = -4, ∂f/∂y = -4 * 1 = -4

**Key patterns:**
- Add gate: distributes upstream gradient equally to all inputs (local gradient = 1)
- Multiply gate: swaps inputs and scales by upstream gradient
- Max gate: routes gradient to whichever input was larger, zero to the other
