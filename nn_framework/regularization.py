#Import numpy so we can perform operations on lists of values e.g. changing certain numbers in a list where a condition is met
import numpy as np

#L1 regulizaer - i.e. Lasso regulizer
#Weights can be eliminated from the model when they hit zero
#The adjustment on all weights is constant
class L1(object):
    def __init__(self, regularization_amount=1e-2):
        #How strong the regularizer will be - out as lamda in machine learning notation
        #1/100th or 1e-2 is the default
        self.regularization_amount = regularization_amount

    def update(self, layer):
        values = layer.weights
        #Set 'delta' as a figure to adjust weights by in each iteration
        delta = self.regularization_amount * layer.learning_rate
        #Use numpy to manipulate all the weight values conditional on what they are
        #Positive weights are reduced by the delta amount, negative weights are increased by the delta regularization_amount
        #Any weights wheir their absolute value is  between zero and delta are set to zero, i.e. removed
        #Across every iteration, if a node and weight is important it is likely to stick in place as delta isn't especially strong
        #Though where nodes and weights are not important to tthe weight is likely to pull down over several iterations, and especially
        #those weights already around zero will likely be removed from the model
        values[np.where(np.abs(values) < delta)] = 0
        values[np.where(values > 0)] -= delta
        values[np.where(values < 0)] += delta
        return values

#L2 regulizaer - i.e. Ridge or Tikhonov regulizer
#Weights are not eliminated from the model as they converge towards zero
#The adjustment on all weights is proportionate to the size of the weights
#Particularly effective at pulling in large weights
class L2(object):
    def __init__(self, regularization_amount=1e-2):
        self.regularization_amount = regularization_amount

    def update(self, layer):
        adjustment = (
            2 * self.regularization_amount *
            layer.learning_rate *
            layer.weights
            )
        return layer.weights - adjustment

#Limit regularizer as custom example. Prevent weights from going over a set limit
#We could also code this through a custom loss function which adds amuch  higher loss for the weights above the limit, but here we use a regulizer as an example
#Be cautious to use custom regulizaers as they can have unintened effects and you might overfit or underfit the model yourself if you don't have a clear rationale
class Limit(object):
    def __init__(self, weight_limit=1):
        self.weight_limit = weight_limit

    def update(self, layer):
        values = layer.weights
        values[np.where(values > self.weight_limit)] = self.weight_limit
        values[np.where(values < -self.weight_limit)] = -self.weight_limit
        return values

#Other regularizers exist which we could add e.g. fit weights after a certain number of iterations to avoid over-running the model
