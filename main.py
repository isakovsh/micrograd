import math 
import random 

# Create Value class 
class Value:
    def __init__(self,data,_children=(),_op=""):
        self.data = data 
        self.grad = 0
        self._previous = set(_children)
        self._op = _op
        self._backward = lambda : None

    def __add__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data + other.data, (self,other),"+")

        def _backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad 
        out._backward = _backward

        return out
    
    def __mul__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data * other.data, (self,other),"*")

        def _backward():
            self.grad += other.data * out.grad 
            other.grad += self.data * out.grad 
        out._backward = _backward

        return out 
    
    def __pow__(self,other):
        assert isinstance(other,(int,float))
        out = Value(self.data**other,(self,),"**other")

        def _backward():
            self.grad += other * self.data ** (other-1) * out.grad 
        
        out._backward = _backward

        return out 
    
    def tanh(self):
        out = Value(math.tanh(self.data),(self,),"tanh")

        def _backward():
            self.grad += (1-out.data**2) * out.grad
        
        out._backward = _backward

        return out 
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
    
    def exp(self):
        out = Value(math.exp(self.data),(self,),"exp")

        def _backward():
            self.grad += out.data * out.grad 

        out._backward = _backward
    
        return out 

    def log(self):
        out = Value(math.log(self),(self,),"log")

        def _backward():
            self._grad += (1/self.data) * out.grad 
        out._backward = _backward

        return out 
    
    def backward(self):

        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._previous:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1
        for v in reversed(topo):
            v._backward()
        
    def __neg__(self):
        return self * -1 
    
    def __truediv__(self,other):
        return self * other** -1 
    
    def __radd__(self,other):
        return self + other 
    
    def __rmull__(self,other):
        return self * other 
    
    def __sub__(self,other):
        return self + (-other)
    
    def __rsub__(self,other):
        return self + (-other)
    
    def __rturediv__(self,other):
        return self ** -1 * other 
    
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


class Neuron:
    """"A single neuoron"""
    def __init__(self,xin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(xin)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self,X):
        act = sum((wi*xi  for wi,xi in zip(self.w,X)),self.b)
        out = act.tanh()
        return out 
    
    def parameters(self):
        return self.w + [self.b]
    

class Layer:
    def __init__(self,nin,nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self,X):
        outs = [n(X) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters() ]
    
class MLP:
    def __init__(self,nin,nouts): # 4, [8,10,8,1] 4 8 8 10 10 8 8 1
        sz = [nin] + nouts
        self.layers = [Layer(sz[i],sz[i+1]) for i in range(len(nouts))]

    def __call__(self,X):
        for layer in self.layers:
            X = layer(X)
        return X 

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    

#----------------- Test -----------------------

xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

n = MLP(3, [4,8,4,1])

# training 

for i in range(50):
    ypreds = [n(x) for x in xs]
    loss = sum((yout-ypred)**2 for yout,ypred in zip(ys,ypreds))

    for p in n.parameters():
            p.grad = 0

    loss.backward()

    for p in n.parameters():
            p.data += -0.01 * p.grad

    print(i,loss.data)



