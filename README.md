# Neural Network from Scratch

A comprehensive implementation of neural networks built from the ground up in Python, featuring both numpy-based forward propagation and a micrograd-style automatic differentiation engine for backpropagation.

## 🎓 Learning Resources

This project was built while learning from two exceptional educators in machine learning:

### **Andrej Karpathy**
- [micrograd](https://github.com/karpathy/micrograd) - A tiny scalar-valued autograd engine
- [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) - YouTube series on building neural networks from scratch
- Inspired the `micrograd.py` implementation and backpropagation architecture

### **Sentdex (Harrison Kinsley)**
- [Neural Networks from Scratch in Python](https://nnfs.io/) - Book and tutorial series
- [YouTube Channel](https://www.youtube.com/user/sentdex) - Comprehensive Python and ML tutorials
- Inspired the `layer_dense.py` and forward pass implementation using numpy

---

## 📁 Project Structure

```
neural-network-v1/
├── micrograd.py              # Scalar autograd engine (backprop)
├── micrograd_nn.py           # Neural network using micrograd
├── layer_dense.py            # Dense layers and activations (numpy)
├── network.py                # Neural network container (forward pass)
├── test1.ipynb               # Jupyter notebook experiments
├── micro_grad.ipynb          # Micrograd visualization notebook
└── README.md                 # This file
```

---

## 🚀 Features

### 1. **Micrograd Autograd Engine** (`micrograd.py`)
A lightweight automatic differentiation framework inspired by Andrej Karpathy's micrograd:

- **Scalar-based computation graph** with automatic gradient tracking
- **Backpropagation** through arbitrary computational graphs
- **Supported operations:**
  - Addition, subtraction, multiplication, division
  - Power functions
  - Tanh activation
  - Automatic gradient accumulation via chain rule

**Example:**
```python
from micrograd import value

a = value(2.0, label='a')
b = value(-3.0, label='b')
c = value(10.0, label='c')
e = a * b; e.label = 'e'
d = e + c; d.label = 'd'
f = value(-2.0, label='f')
L = d * f; L.label = 'L'

# Compute all gradients
L.backward()
print(f"dL/da = {a.grad}")  # How much does 'a' affect the loss?
```

### 2. **Dense Layers with NumPy** (`layer_dense.py`)
Efficient vectorized implementations for forward propagation:

**Layers:**
- `layer_dense` - Fully connected layer with weights and biases
- `Ac_relu` - ReLU activation (max(0, x))
- `Ac_softmax` - Softmax activation for classification
- `Ac_tanh` - Tanh activation
- `loss_CategoricalCrossEntropy` - Cross-entropy loss for classification

**Example:**
```python
import layer_dense as ld
import numpy as np

# Create a dense layer: 2 inputs -> 5 neurons
layer1 = ld.layer_dense(2, 5)
activation1 = ld.Ac_relu()

# Forward pass
X = np.random.randn(100, 2)  # 100 samples, 2 features
output = layer1.forward_pass(X)
activated = activation1.forward_pass(output)
```

### 3. **Neural Network Container** (`network.py`)
Sequential model container for building multi-layer networks:

**Example:**
```python
from network import NeuralNetwork
import layer_dense as ld

nn = NeuralNetwork()
nn.add(ld.layer_dense(2, 64))    # Input: 2 features -> 64 neurons
nn.add(ld.Ac_relu())              # ReLU activation
nn.add(ld.layer_dense(64, 64))   # Hidden layer
nn.add(ld.Ac_relu())
nn.add(ld.layer_dense(64, 3))    # Output: 3 classes
nn.add(ld.Ac_softmax())           # Softmax for probabilities

# Forward pass
predictions = nn.forward(X)
```

### 4. **Micrograd Neural Network** (`micrograd_nn.py`)
Neural network implementation using the micrograd autograd engine:

**Classes:**
- `DenseMG` - Dense layer with scalar Value objects for each weight/bias
- `ReLUMG` - ReLU activation with automatic differentiation
- `NeuralNetworkMG` - Network container with backprop support

**Features:**
- Automatic gradient computation through backpropagation
- SGD optimizer with learning rate
- MSE loss function

**Training Example:**
```python
from micrograd_nn import NeuralNetworkMG, DenseMG, ReLUMG, mse_loss
import numpy as np

# Create network
nn = NeuralNetworkMG()
nn.add(DenseMG(2, 8), ReLUMG())
nn.add(DenseMG(8, 3), None)

# Training loop (per-sample)
for epoch in range(100):
    for i in range(len(X)):
        # Forward pass
        outputs = nn.forward_sample(X[i])
        loss = mse_loss(outputs, Y[i])  # Y is one-hot encoded
        
        # Backward pass
        nn.zero_grads()
        loss.backward()
        
        # Update weights
        nn.step_sgd(lr=0.01)
```

---

## 📊 Jupyter Notebooks

### `test1.ipynb`
Interactive experiments with:
- Single neuron forward pass calculations
- Layer-wise computations
- Spiral dataset visualization using matplotlib
- One-hot encoding with numpy
- Categorical cross-entropy loss
- Full network training pipeline

### `micro_grad.ipynb`
Visualization and experimentation with micrograd:
- Computational graph construction
- Graphviz visualization of operations
- Manual gradient calculations
- Backward propagation walkthrough

---

## 🛠️ Installation

### Requirements
```bash
pip install numpy matplotlib graphviz nnfs
```

### Optional (for visualization)
Install Graphviz system package:
- **Windows:** Download from [graphviz.org](https://graphviz.org/download/)
- **Linux:** `sudo apt-get install graphviz`
- **macOS:** `brew install graphviz`

---

## 📚 Key Concepts Implemented

### 1. **Forward Propagation**
- Matrix multiplication for efficient batch processing
- Activation functions (ReLU, Softmax, Tanh)
- Layer stacking and composition

### 2. **Backpropagation**
- Chain rule for gradient computation
- Topological sorting of computation graph
- Gradient accumulation for nodes used multiple times

### 3. **Autograd Magic**
- Dynamic computation graph construction
- Automatic gradient tracking through operations
- Lazy gradient computation (only when `.backward()` is called)

### 4. **Neural Network Architecture**
```
Input Layer (features)
    ↓
Dense Layer + ReLU
    ↓
Dense Layer + ReLU
    ↓
Dense Layer + Softmax
    ↓
Output (class probabilities)
```

---

## 🎯 Example: Training on Spiral Dataset

```python
import nnfs
from nnfs.datasets import spiral_data
from network import NeuralNetwork
import layer_dense as ld

# Generate spiral dataset
nnfs.init()
X, y = spiral_data(100, 3)  # 100 samples per class, 3 classes

# Build network
nn = NeuralNetwork()
nn.add(ld.layer_dense(2, 64))
nn.add(ld.Ac_relu())
nn.add(ld.layer_dense(64, 3))
nn.add(ld.Ac_softmax())

# Forward pass
predictions = nn.forward(X)

# Calculate loss
loss_fn = ld.loss_CategoricalCrossEntropy()
loss = loss_fn.calculate_loss(predictions, y, num_of_classes=3)
print(f"Loss: {loss}")
```

---

## 🔬 What Makes This Different?

1. **Built from scratch** - No high-level frameworks like TensorFlow or PyTorch
2. **Educational focus** - Clear, commented code showing how everything works
3. **Dual implementation:**
   - NumPy for efficient forward propagation
   - Micrograd for understanding backpropagation
4. **Visualization** - Graphviz integration to see computation graphs
5. **Interactive** - Jupyter notebooks for experimentation

---

## 🧠 Learning Outcomes

By building this project, you learn:

- ✅ How matrix operations power neural networks
- ✅ The math behind backpropagation and chain rule
- ✅ How automatic differentiation engines work
- ✅ Building computational graphs dynamically
- ✅ Weight initialization and gradient descent
- ✅ Activation functions and their purposes
- ✅ Loss functions for classification
- ✅ One-hot encoding and categorical data
- ✅ Vectorization for performance

---

## 📈 Future Improvements

- [ ] Add more activation functions (LeakyReLU, ELU, Swish)
- [ ] Implement batch normalization
- [ ] Add optimizers (Adam, RMSprop, Momentum)
- [ ] Convolutional layers for image processing
- [ ] Dropout for regularization
- [ ] Learning rate scheduling
- [ ] Mini-batch gradient descent
- [ ] Model serialization (save/load weights)
- [ ] GPU acceleration with CuPy
- [ ] Advanced loss functions (Focal loss, etc.)

---

## 🤝 Acknowledgments

Huge thanks to:

- **[Andrej Karpathy](https://karpathy.ai/)** for making neural networks approachable and inspiring the autograd implementation
- **[Sentdex (Harrison Kinsley)](https://pythonprogramming.net/)** for clear explanations of neural network fundamentals and the numpy-based approach

Their teaching made this project possible! 🙌

---

## 📝 License

This project is for educational purposes. Feel free to use and modify for learning!

---

## 💡 Tips for Learners

1. **Start with micrograd.py** - Understand how autograd works at the scalar level
2. **Visualize the graphs** - Use the graphviz functions to see how operations connect
3. **Run the notebooks** - Interactive experimentation is key to understanding
4. **Modify and break things** - Change hyperparameters, architectures, and see what happens
5. **Compare implementations** - See how micrograd_nn.py differs from layer_dense.py
6. **Read the comments** - The code is heavily commented to explain the "why"

---

**Happy Learning! 🚀🧠**

*"The best way to understand deep learning is to build it yourself."*
