import math 

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
            self.grad += out.grad * other.value 
            other.grad += out.grad * self.value

        out._backward = _backward
    
    def __pow__(self,other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,)) 

        def _backward():
            self.grad += (other * self.data**(other -1 ) * out.grad) 

        out._backward = _backward

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        out = Value(math.tanh(self.data),(self,),"tanh")
        def _backward():
            self.backward = (1-out**2) * out.grad
        out.backward = _backward
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
    def __repr__(self) -> str:
        return self.__str__()
