
class BaseOptimizer:

    def __init__(self,model_parameters,lr=0.001):
        self.model_parameters = model_parameters
        self.lr = lr

    def zero_grad(self):
        for param in self.model_parameters:
            param.grad = 0


class SGD(BaseOptimizer):
    def __init__(self, model_parameters, lr=0.001,momentum=0):
        super().__init__(model_parameters, lr)
        self.momentum = momentum 
        self.velocities = {id(param): 0 for param in model_parameters}

    def step(self):
        for param in self.model_parameters:
            if param is None:
                continue 

            velocity = self.momentum * self.velocities[id(param)] - self.lr * param.grad
            self.velocities[id(param)] = velocity

            param.data += velocity
            

