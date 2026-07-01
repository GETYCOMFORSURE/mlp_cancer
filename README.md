# MLP for breast cancer classification

A 1-hidden-layer neural network built in numpy

## Files
- `learning.md` — notes from 3B1B and CS231n
- `derive.md` — derived backpropagation math
- `data_loader.py` — load, normalize, shuffle, split, mini-batch generator
- `model.py` — forward pass, binary cross-entropy loss, backpropagation, gradient descent
- `train.py` — training loop and evaluation function
- `benchmark.py` — comparison against sklearn SVM

## Results
| Model | Test Accuracy |
|-------|--------------|
| MLP (this implementation) | 97.8% |
| SVM with RBF kernel (sklearn) | 97.4% |

## Stack
Python, NumPy, scikit-learn (data loading and benchmark only)
