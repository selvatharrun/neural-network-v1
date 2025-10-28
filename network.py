import numpy as np
import layer_dense as ld
import nnfs

from nnfs.datasets import spiral_data


nnfs.init()  # Initialize nnfs (sets random seed, etc.)

class NeuralNetwork:
    """Very small container for layers with a forward method.

    Usage:
        nn = NeuralNetwork()
        nn.add(layer)
        out = nn.forward(X)
    """
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, inputs):
        output = inputs
        for layer in self.layers:
            # Each layer must implement forward_pass(inputs) -> outputs
            output = layer.forward_pass(output)
        return output


if __name__ == '__main__':
    # Quick smoke test using random data so it doesn't depend on nnfs
    X,Y = spiral_data(100,3)
    nn = NeuralNetwork()
    nn.add(ld.layer_dense(2, 5))
    nn.add(ld.Ac_relu())
    nn.add(ld.layer_dense(5, 3))
    nn.add(ld.Ac_softmax())

    out = nn.forward(X)
    print('out.shape =', out.shape)
    print('first row =', out[0])
