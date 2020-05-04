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

        self.learning_rate = .05

        # Choose random weights.
        # Inputs match to rows. Outputs match to columns.
        # Add one to m_inputs to account for the bias term.
        # self.initial_weight_scale = 1 / self.m_inputs
        self.initial_weight_scale = 1
        self.weights = self.initial_weight_scale * (np.random.sample(
            size=(self.m_inputs + 1, self.n_outputs)) * 2  - 1)
        self.w_grad = np.zeros((self.m_inputs + 1, self.n_outputs))
        self.x = np.zeros((1, self.m_inputs + 1))
        self.y = np.zeros((1, self.n_outputs))


    def forward_prop(self, inputs):
        """
        Propagate the inputs forward through the network.
        inputs: 2D array
            One column array of input values.
        """
        bias = np.ones((1, 1))
        self.x = np.concatenate((inputs, bias), axis=1)
        v = self.x @ self.weights
        self.y = self.activate.calc(v)
        return self.y

    def back_prop(self, de_dy):
        """
        Propagate the outputs back through the layer.
        """
        #Find the derivative of the output y with respect to the linear output in the node v
        #v is the output from input vector x multiplied by the weights vector, before it goes
        #through the activation function. So v is like an intermediate output wrt the acitvation
        #function. Hence, the derivative here is the derivative of the activation function, calc(d)
        dy_dv = self.activate.calc_d(self.y)
        #Self.weights.transpose() = dy/dx - use the chain rule to find de_dx from de_dy, dy_dv and dv_dx 
        de_dx = (de_dy * dy_dv) @ self.weights.transpose()
        #Return derivatives at all layer expect the bias layer (the -1 excludes this)
        return de_dx[:, :-1]
