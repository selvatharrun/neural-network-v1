import numpy as np
class layer_dense:
    # num_of_neurons is how many neurons we want in this layer.
    def __init__(self, input_size, num_of_neurons):
        self.weights = 0.1*np.random.randn(input_size, num_of_neurons) #i want it between -0.1 and 0.1 initially
        self.biases = np.zeros((1,num_of_neurons))
    def forward_pass(self, inputs):
        print(inputs.shape, " " , self.weights.shape )
        self.outputs = np.dot(inputs,self.weights) + self.biases
        return self.outputs
        
class Ac_relu:
    def forward_pass(self, inputs):
        # ReLU activation: keep positive values, set negatives to 0
        self.outputs = np.maximum(0, inputs)
        return self.outputs