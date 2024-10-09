from value import Value 

class MSE:

    def __init__(self) -> None:
        pass

    def __call__(self,y,y_pred):
        loss = sum((yi-ypredi)**2 for yi,ypredi in zip(y,y_pred))
        loss *=     1/len(y)
        return loss 
    

class BinaryCrossEntropy:

    def __init__(self) -> None:
        pass

    def __call__(self,y,ypred):
        
        loss = sum((yi*ypredi.log()) + (Value(1.0) - yi)*(Value(1)-ypredi).log() for yi,ypredi in zip(y,ypred))
        loss *= -1/len(y)
        return Value(loss)
        



        
        