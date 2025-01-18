import math
from engine import Value 


def sigmoid(value):
    out = Value(1 / (1 + math.exp(-value.data)), (value,))
    def _backward():
        out.grad = out.data * (1 - out.data) * out.grad
        value.grad += out.grad
    out._backward = _backward
    return out

def relu(value):
    out = Value(max(0,value.data))
    def _backward():
        value.grad += (1.0 if out.data > 0 else 0.0) * out.grad
    out._backward = _backward
    return out
    
def tanh(value):
    out = Value(math.tanh(value.data),(value,),"tanh")
    def _backward():
        value.backward += (1-out.data**2) * out.grad
    out.backward = _backward
    return out