import numpy as np
from micrograd import value
import math

class DenseHybrid:
    """Dense layer that stores weights/biases as numpy arrays but integrates with micrograd.Value
    by creating scalar Value outputs and registering backward closures that accumulate gradients
    into numpy dW/db arrays.
    """
    def __init__(self, input_size, num_neurons):
        self.input_size = input_size
        self.num_neurons = num_neurons
        # store weights as (num_neurons, input_size) for easier per-neuron dot
        self.W = 0.1 * np.random.randn(num_neurons, input_size)
        self.b = np.zeros(num_neurons)
        # gradient accumulators
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward_sample(self, inputs):
        # inputs: list of micrograd.Value objects
        outs = []
        xs = np.array([inp.data for inp in inputs])
        for n in range(self.num_neurons):
            s_val = float(self.W[n].dot(xs) + self.b[n])
            # create Value whose children are the input Value objects so backprop can flow
            out = value(s_val, children=tuple(inputs), op='dense')

            def _backward(out=out, n=n, inputs=inputs):
                # accumulate weight and bias gradients (numpy)
                g = out.grad
                for j, inp in enumerate(inputs):
                    # dW = x_j * grad_out
                    self.dW[n, j] += inp.data * g
                    # propagate gradient to input Values: dL/dx_j += w_j * grad_out
                    inp.grad += self.W[n, j] * g
                self.db[n] += g

            out._backward = _backward
            outs.append(out)
        return outs

    def zero_grads(self):
        self.dW.fill(0.0)
        self.db.fill(0.0)

    def step_sgd(self, lr=0.01):
        # apply accumulated gradients to numpy weights/biases
        self.W -= lr * self.dW
        self.b -= lr * self.db

class ReLUMG:
    def forward(self, values):
        out = []
        for v in values:
            # implement ReLU with autograd
            o = value(v.data if v.data > 0 else 0.0, children=(v,), op='relu')
            def _back(v=v, o=o):
                v.grad += (1.0 if o.data > 0 else 0.0) * o.grad
            o._backward = _back
            out.append(o)
        return out

class NeuralNetworkMG:
    def __init__(self):
        self.layers = []
        self.activations = []

    def add(self, layer, activation=None):
        self.layers.append(layer)
        self.activations.append(activation)

    def forward_sample(self, x):
        # x: iterable of floats for a single sample
        values = [value(xi) for xi in x]
        for layer, act in zip(self.layers, self.activations):
            # pass Value objects to layers so they can create proper children links
            values = layer.forward_sample(values)
            if act is not None:
                values = act.forward(values)
        return values

    def parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                yield from layer.parameters()

    def zero_grads(self):
        for layer in self.layers:
            if hasattr(layer, 'zero_grads'):
                layer.zero_grads()
            elif hasattr(layer, 'parameters'):
                for p in layer.parameters():
                    p.grad = 0.0

    def step_sgd(self, lr=0.01):
        for layer in self.layers:
            if hasattr(layer, 'step_sgd'):
                layer.step_sgd(lr)
            elif hasattr(layer, 'parameters'):
                for p in layer.parameters():
                    p.data -= lr * p.grad

# Simple MSE loss between output Values and target one-hot (list of floats)
def mse_loss(outputs, target):
    # outputs: list of Value
    assert len(outputs) == len(target)
    s = value(0.0)
    for o, t in zip(outputs, target):
        diff = o - value(t)
        s = s + (diff * diff)
    return s * (1.0 / len(outputs))

# small smoke test function
if __name__ == '__main__':
    import nnfs
    from nnfs.datasets import spiral_data
    nnfs.init()
    X,y = spiral_data(10,3)
    # one-hot
    num_classes = np.max(y) + 1
    Y = np.eye(num_classes)[y]

    nn = NeuralNetworkMG()
    # use numpy-backed DenseHybrid (simpler weights) in the smoke test
    nn.add(DenseHybrid(2, 8), ReLUMG())
    nn.add(DenseHybrid(8, 3), None)

    # forward/backward on first sample
    x0 = X[0]
    t0 = Y[0]
    out_vals = nn.forward_sample(x0)
    loss = mse_loss(out_vals, t0)
    # set gradient seed and backprop
    loss.grad = 1.0
    loss.backward()
    print('loss data:', loss.data)
    # perform one SGD step
    nn.step_sgd(lr=0.1)
    nn.zero_grads()
    print('SGD step done')
