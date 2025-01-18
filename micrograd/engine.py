import math 
import random

class Value:
    def __init__(self,data,_children=()):
        self.data = data 
        self.grad = 0 
        self._backward = lambda : None
        self._prev = set(_children)
    
    def __add__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data + other.data,(self,other)) 
        
        def _backward():
            self.grad += 1.0 * out.grad 
            other.grad += 1.0 * out.grad 
        
        out._backward = _backward

        return out 

    def __mul__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data * other.data ,(self,other))

        def _backward():
            self.grad += out.grad * other.data 
            other.grad += out.grad * self.data

        out._backward = _backward
        return out
    
    def __pow__(self,other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,)) 

        def _backward():
            self.grad += (other * self.data**(other -1 ) * out.grad) 

        out._backward = _backward
        
        return out

    def relu(self):
        out = Value(max(0,self.data))
        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad
            
        out._backward = _backward
        return out
    
    def tanh(self):
        out = Value(math.tanh(self.data),(self,))
        def _backward():
            self.grad += (1-out.data**2) * out.grad
        out._backward = _backward
        return out
    
    def log(self):
        if self.data <= 0:
            self.data = 1e-17 #Not great but temporary solution to having values of 0.0
        out = Value(math.log(self.data), (self,))
        
        def _backward():
            self.grad += (1.0 / self.data) * out.grad
            self._backward()
        out.backward = _backward
        
        return out

    def backward(self):

        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1
    def __radd__(self, other): # other + self
        return self + other
    def __sub__(self, other): # self - other
        return self + (-other)
    def __rsub__(self, other): # other - self
        return other + (-self)
    def __rmul__(self, other): # other * self
        return self * other
    def __truediv__(self, other): # self / other
        return self * other**-1
    def __rtruediv__(self, other): # other / self
        return other * self**-1
        
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


class Neuron:

    def __init__(self,nin):
        self.w = [Value(random.gauss(0, math.sqrt(2 / nin))) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))
    
    def __call__(self,X):
        out = sum((wi*Xi for wi, Xi in zip(self.w,X)),self.b)
        return out.tanh()
    
    def parametrs(self):
        return self.w + [self.b]

class Layer:
    
    def __init__(self,nin,nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self,X):
        outs = [n(X) for n in self.neurons]
        return outs[0] if len(outs)==1 else outs 
    
    def parametrs(self):
        return [p for neroun in self.neurons for p in neroun.parametrs()]

class MLP:

    def __init__(self, nin,nouts):
        sz = [nin] + nouts 
        self.layers = [Layer(sz[i],sz[i+1]) for i in range(len(nouts))]
        
    def __call__(self,X):
        for layer in self.layers:
            X = layer(X)
        return X

    def parametrs(self):
        return [p for layer in self.layers for p in layer.parametrs()]
