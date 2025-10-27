import numpy as np
import layer_dense as ld


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


class ActivationSoftmax:
    def forward_pass(self, inputs):
        # Numerically stable softmax
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.outputs = probabilities
        return self.outputs


if __name__ == '__main__':
    # Quick smoke test using random data so it doesn't depend on nnfs
    X = np.random.randn(10, 2)
    nn = NeuralNetwork()
    nn.add(ld.layer_dense(2, 5))
    nn.add(ld.Ac_relu())
    nn.add(ld.layer_dense(5, 3))
    nn.add(ActivationSoftmax())

    out = nn.forward(X)
    print('out.shape =', out.shape)
    print('first row =', out[0])
