import numpy as np
import math


e = math.e

#we will be building micrograd from scratch
class value:
    def __init__(self, data, children = (), op='', label =""):
        self.data = data
        self.children = set(children)
        self.grad = 0 # grad is 0 by default
        self._backward = lambda : None #empty lambda function by default
        self.op = op
        self.label = label
    
    def __add__(self,other):
        out = value(self.data + other.data, children=(self, other), op='+')
        # _backward primarily talks about how the gradient would flow back.
        def _backward():
            #basically chain rule, lets say for example:
            # there is L -> Z -> X+Y 
            # now how tweaking x or y is going to affect L depends on
            # ∂L/∂z * ∂z/∂x = ∂L/∂x this is basically simple chain rule, and we know that 
            # ∂z/∂x = 1. by derivative
            # (f(x+h) - f(x)) / h 
            self.grad += out.grad * 1 #we do += to coz we want the gradient to accumulate when the same node is used more than once,
            #instead of fucking replacing it, like we were doing before.
            other.grad += out.grad * 1
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + (other if isinstance(other, value) else value(other))

    def __neg__(self):
        out = value(-self.data, children=(self,), op='neg')
        def _backward():
            self.grad += -out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, value) else value(other)
        return self + (-other)

    def __rsub__(self, other):
        other = other if isinstance(other, value) else value(other)
        return other - self
    
    def __mul__(self, other):
        other = other if isinstance(other, value) else value(other) #to still continue to work if the other shit isnt a value object and a normal number.
        out = value(self.data * other.data, (self, other), '*')

        def _backward():
            # same chain rule bs
            self.grad += other.data * out.grad #the derivative would return eh other.data u understand the rest.
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * (other if isinstance(other, value) else value(other))

    def __truediv__(self, other):
        other = other if isinstance(other, value) else value(other)
        return self * (other ** -1)

    def __pow__(self, exponent):
        # exponent is a number
        out = value(self.data ** exponent, (self,), f'**{exponent}')
        def _backward():
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad
        out._backward = _backward
        return out
    def tanh(self):
        out = value (( np.float_power(e , self.data) - np.float_power(e , -self.data) ) / (np.float_power(e , self.data) + np.float_power(e , -self.data) ), children=(self,), op='tanh')
        def _backward():
            self.grad += (1 - out.data**2) * out.data
        out._backward = _backward
        return out

    def backward(self):
        #topological order all of the children in the graph
        topo = []
        visited = set()

        def build_topo(root):
            if root not in visited:
                visited.add(root)
                for child in root.children:
                    build_topo(child)
                topo.append(root)

        # ultimate goal is to check how much a nudge of a node would affect the result.
        # this wil give gradient to all the nodes. this will apply chain rule to all the nodes.
        # notice the difference of _backward and backward. this is beautiful
        build_topo(self)
        self.grad = 1
        
        for node in reversed(topo):
            node._backward()
    
    