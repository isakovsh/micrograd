from .engine import Value 

class MSELoss:
    
    def __call__(self,y_true, y_pred):
        """
        Compute the mean squared error loss:
        y_pred: List of predicted probabilities (softmax output)
        y: List of true one-hot labels
        """
        loss = Value(0.0)
        
        n = len(y_true)

        for i in range(n):
            loss += (y_true[i]-y_pred[i])**2
        
        loss = loss * Value(1/n)
        loss.grad = 1
        return loss
    
class CrossEntropyLoss:

    def __call__(self, y_pred, y):
        """
        Compute the cross-entropy loss:
        y_pred: List of predicted probabilities (softmax output)
        y: List of true one-hot labels
        """
        loss = Value(0.0)
        n = len(y)
        # assert all(len(yi.data) == len(y_predi) for yi, y_predi in zip(y, y_pred)), "Mismatch in class sizes"

        # Compute the cross-entropy loss
        losses = []
        for yi, y_predi in zip(y, y_pred):
            loss = sum(-yij * y_predij.log() for yij, y_predij in zip(yi, y_predi))
            losses.append(loss)

        return sum(losses) / Value(n)