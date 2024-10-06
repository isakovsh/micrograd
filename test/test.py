import random
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from value import Value
from nn import Layer, Neuron, MLP

np.random.seed(1337)
random.seed(1337)

from sklearn.datasets import make_moons, make_blobs
X, y = make_moons(n_samples=100, noise=0.1)

y = y*2 - 1 # make y be -1 or 1
# visualize in 2D
# plt.figure(figsize=(5,5))
# plt.scatter(X[:,0], X[:,1], c=y, s=20, cmap='jet')

model = MLP(2,[8,16,8,1])

for k in range(50):
    #forwardpass 
    inputs = [list(map(Value, xrow)) for xrow in X]
    ypreds = list(map(model, inputs))
    #loss compute 
    loss = sum((ypred-yi)**2 for ypred,yi in zip(ypreds,y))
    
    #backward 
    for p in model.parameters():
        p.grad = 0
    
    loss.backward()
    
    for p in model.parameters():
        p += -0.01 * p.grad

    if k%10 == 0:
        print(k,loss.data)

