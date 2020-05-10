# -*- coding: utf-8 -*-
import numpy as np

class Dense(object):
    def __init__(
        self,
        m_inputs,
        n_outputs,
        activate,
        debug=False,
    ):
        self.debug = debug
        self.m_inputs = int(m_inputs)
        self.n_outputs = int(n_outputs)
        self.activate = activate

        #The learning rate is a hyperparameter we can adjust in the model. It sets the rate or 'jumps' at which the model
        #decreases the size of the error or 'loss function' at each iteration. Too big and the model will jump erratically, too small and the model will take longer to converge on a minimum
        self.learning_rate = .001

        # Choose random weights.
        # Inputs match to rows. Outputs match to columns.
        # Add one to m_inputs to account for the bias term.
        # self.initial_weight_scale = 1 / self.m_inputs
        self.initial_weight_scale = 1
        #Randomise the weights as a start point for the model to work from  these will likely be inaccurate
        self.weights = self.initial_weight_scale * (np.random.sample(
            size=(self.m_inputs + 1, self.n_outputs)) * 2  - 1)
        self.w_grad = np.zeros((self.m_inputs + 1, self.n_outputs))
        self.x = np.zeros((1, self.m_inputs + 1))
        self.y = np.zeros((1, self.n_outputs))

        self.regularizers = []

    #Method to add a specified regularizer to the layer
    def add_regularizer(self, new_regularizer):
        self.regularizers.append(new_regularizer)

    def forward_prop(self, inputs):
        """
        Propagate the inputs forward through the network.
        inputs: 2D array
            One column array of input values.
        """
        #Set the bias term - the b0 or 'intercept of the model' which is a vector of ones (??)
        bias = np.ones((1, 1))
        #Create the full set of model inputs by combining the inputs x with the bias (Axis=1 ??)
        self.x = np.concatenate((inputs, bias), axis=1)
        #Find the linear output v by matrix multiplying the inputs by the weights
        v = self.x @ self.weights
        #Apply the activation function to the linear output v to make the nonlinear output y
        self.y = self.activate.calc(v)
        #Return y as the ouptput for the model, and endpoint of the forward propagation
        return self.y

    def back_prop(self, de_dy):
        """
        Propagate the outputs back through the layer.
        """
        #Find the derivative of the output y with respect to the linear output in the node v
        #v is the linear output from input vector x multiplied by the weights vector, before it goes
        #through the activation function. So v is like an intermediate output wrt the acitvation
        #function. Hence, the derivative here is the derivative of the activation function, calc(d)
        dy_dv = self.activate.calc_d(self.y)
        #v = self.x @ self.weights - the inputs (matrix) multiplied by the weights - the derivative of v with
        #respect to one of these components is the other
        #dv_dw = self.x
        #dv_dx = self.weights

        #Sensitity of the output y with respect to weights - dy_dw = dv_dw * dy_dv - transpose is so the matrices can
        #multipy with m x n and n x p rule
        dy_dw = self.x.transpose() @ dy_dv
        #Sensitivity of error with respect to weights, use to change weights and move down the loss function
        de_dw = de_dy * dy_dw
        self.weights -= de_dw * self.learning_rate

        #Add in regularization at this layer
        #For each regulizer in the layer's list of regularizers
        for regulizer in self.regularizers:
            #Regulate the weights and prevent these moving too far so as to avoid overfitting i.e. the issue where the model fits predicitons to 'noise' as well as the signal.
            self.weights = regulizer.update(self)

        #Self.weights.transpose() = dy/dx - use the chain rule to find de_dx from de_dy, dy_dv and dv_dx
        de_dx = (de_dy * dy_dv) @ self.weights.transpose()
        #Return derivatives at all layer expect the bias layer (the -1 excludes this)
        return de_dx[:, :-1]
