import numpy as np

# All of these need to be able to handle 2D numpy arrays as inputs.
class tanh(object):
    #A static method means the class is not instantianted and is treated as a function or method
    #A public static class works similarly in C#
    @staticmethod
    def calc(v):
        return np.tanh(v)

    @staticmethod
    def calc_d(v):
        return 1 - np.tanh(v) ** 2

#??? Why is this def instead of class?
def logistic(v):
    @staticmethod
    def calc(v):
        return 1 / (1 + np.exp(-v))

    @staticmethod
    def calc_d(v):
        return calc(v) * (1 - calc(v))


def relu(v):
    @staticmethod
    def calc(v):
        return np.maximum(0, v)

    @staticmethod
    def calc_d(v):
        derivative = 0
        if v > 0:
            derivative = 1
        return derivative
