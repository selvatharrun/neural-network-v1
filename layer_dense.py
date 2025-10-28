import numpy as np
import math

euler_num = math.e

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
    
class Ac_softmax:
    def forward_pass(self,inputs):
        exp = np.float_power(euler_num, inputs - np.max(inputs, axis=1, keepdims=True))
        #deduce the maximum value for numerical stability
        #axis and keepdims to maintain the original shape for broadcasting
        sum1 = np.sum(exp, axis=1, keepdims=True)   
        self.outputs = np.divide(exp, sum1)
        return self.outputs

class Ac_tanh:
    def forward_pass(self, inputs):
        self.outputs = np.divide( np.float_power(euler_num, inputs) - np.float_power(euler_num, -inputs) ,np.float_power(euler_num, inputs) + np.float_power(euler_num, -inputs))
        return self.outputs

class loss_CategoricalCrossEntropy:
    def calculate_loss(self, output, y):
        # support sparse labels (class indices) and one-hot encoded labels
        eps = 1e-15
        output_clipped = np.clip(output, eps, 1 - eps)
        if y.ndim == 1:
            correct_confidences = output_clipped[np.arange(len(output_clipped)), y]
        else:
            correct_confidences = np.sum(output_clipped * y, axis=1)
        return -np.mean(np.log(correct_confidences))