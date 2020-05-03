import numpy as np

# All of these expect two identically sized numpy arrays as inputs
# and return the same size error output.


class sqr(object):
    @staticmethod
    #Squared loss function
    def calc(x, y):
        return (y - x)**2

    #The derivative of the sum squared error with respect to changes in y
    @staticmethod
    def calc_d(x, y):
        return 2 * (y - x)
