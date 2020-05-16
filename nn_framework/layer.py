import numpy as np

#A generic layer which can act as a starting point for custom layers
#Contains all essential functions and an initialiser for other layer classes to inherit
class GenericLayer(object):

    def __init__(self, previous_layer):
        self.previous_layer = previous_layer
        #Set number of nodes using the size of the output previous layer (which was set at the start of the model
        #using the list of the nodes in the whole model - to the previous layer was set bearing in mind the next layer too)
        self.size = self.previous_layer.y.size
        self.reset()

    #Reset is called before each forward and backward pass
    def reset(self):
        #Set the input and output vectors and their derivates up as 1-dimensional null vectors
        #that are the size of the number of nodes.
        self.x = np.zeros((1,self.size))
        self.y = np.zeros((1,self.size))
        #Do the same for the derivatives - these accumulate if multiple layers contribute to the gradient
        self.de_dx = np.zeros((1,self.size))
        self.de_dy = np.zeros((1,self.size))

    #Whatever other keyword arguments are passed in, these are passed into **kwargs
    #This is to keep the function working should we expand it in our custom layers
    def forward_pass(self, **kwargs):
        self.x += self.previous_layer.y
        self.y = self.x

    #Pass back the derivative of the error with respect to this layer's input to the previous layer
    def backward_pass(self):
        self.de_dx = self.de_dy
        self.previous_layer.de_dy += self.de_dx


#Create the dense layer which inherits from GenericLayer
class Dense(GenericLayer):
    def __init__(
        self,
        n_outputs,
        activation_function,
        previous_layer = None,
        dropout_rate=0,
    ):
        self.previous_layer = previous_layer
        self.m_inputs = self.previous_layer.y.size
        self.n_outputs = int(n_outputs)
        self.activation_function = activation_function
        self.dropout_rate = dropout_rate

        #The learning rate is a hyperparameter we can adjust in the model. It sets the rate or 'jumps' at which the model
        #decreases the size of the error or 'loss function' at each iteration. Too big and the model will jump erratically, too small and the model will take longer to converge on a minimum
        self.learning_rate = .001

        #Choose random weights. Inputs match to rows with an addition for the bias terms,
        #outputs match to columns
        self.weights = (np.random.sample(
            size=(self.m_inputs + 1, self.n_outputs)) * 2 - 1)

        #Reset the layer
        self.reset()
        self.regularizers = []


    #Method to add a specified regularizer to the layer
    def add_regularizer(self, new_regularizer):
        self.regularizers.append(new_regularizer)

    def reset(self):
        self.x = np.zeros((1,self.m_inputs))
        self.y = np.zeros((1,self.n_outputs))
        self.de_dx = np.zeros((1,self.m_inputs))
        self.de_dy = np.zeros((1,self.n_outputs))

    def forward_pass(self,  evaluating=False, **kwargs):
        """
        Propagate the inputs forward through the network.
        inputs: 2D array
            One column array of input values.
        """

        #Propagate the previous layer output if there is a previous layer
        if self.previous_layer is not None:
            self.x += self.previous_layer.y

        #Apply dropout, only during training runs
        if evaluating:
            dropout_rate = 0
        else:
            #Adopt this layer's dropout rate
            dropout_rate = self.dropout_rate

        #Create an array of boolean variables the length of the nodes
        self.i_dropout = np.zeros(self.x.size, dtype=bool)
        #Generate a number between 0 and 1 on a uniform distribution for each node.
        #Where the value generated is less than the dropout rate, set the boolean value
        #to True in the array we created. Where it exceeds the dropout rate, set the boolean to false
        #If say the dropout rate is .2 we would expect on average that 20% of nodes are set to True and drop out
        self.i_dropout[np.where(
            np.random.uniform(size=self.x.size) < dropout_rate)] = True
        #Use the array of booleans on the input vector x. Where i_dropout is True, set x to zero so it is removed from the model
        self.x[:, self.i_dropout] = 0
        #Where, by contrast, i_dropout is False, multiply x through so that the outputs of the model are at a scale equivalent to what might expect if we were running the model without dropout. The expected value and variance, the distributional properties of the model's outputs, should remain similar.
        #If the dropout rate is 0.2 then outputs in the remainder of nodes are amplified by 1/(1-0.2) = 1.25 or if the droupout is 0.5 then the remainder of nodes are amplified by 1/(1-0.5) = 2
        self.x[:, np.logical_not(self.i_dropout)] *= 1 / (1 - dropout_rate)

        #Set the bias term - the b0 or 'intercept of the model' which is a vector of ones (??)
        bias = np.ones((1, 1))
        #Create the full set of model inputs by combining the inputs x with the bias (Axis=1 ??)
        #Not a fan of using the x_with_bias local variable, I'd stick with self.x unless there's a reason ??
        x_with_bias = np.concatenate((self.x, bias), axis=1)
        #Find the linear output v by matrix multiplying the inputs by the weights
        v = x_with_bias @ self.weights
        #Apply the activation function to the linear output v to make the nonlinear output y
        self.y = self.activation_function.calc(v)
        #Return y as the ouptput for the model, and endpoint of the forward propagation
        return self.y

    def backward_pass(self):
        #Propagate the outputs back through the layer.
        bias = np.ones((1,1))
        x_with_bias = np.concatenate((self.x, bias), axis=1) #?? what does axis = 1 do

        #Find the derivative of the output y with respect to the linear output in the node v
        #v is the linear output from input vector x multiplied by the weights vector, before it goes
        #through the activation function. So v is like an intermediate output wrt the acitvation
        #function. Hence, the derivative here is the derivative of the activation function, calc(d)
        dy_dv = self.activation_function.calc_d(self.y)
        #v = self.x @ self.weights - the inputs (matrix) multiplied by the weights - the derivative of v with
        #respect to one of these components is the other
        #dv_dw = self.x
        #dv_dx = self.weights


        #Sensitity of the output y with respect to weights - dy_dw = dv_dw * dy_dv - transpose is so the matrices can
        #multipy with m x n and n x p rule
        dv_dw = x_with_bias.transpose()
        dv_dx = self.weights.transpose()

        dy_dw = dv_dw @ dy_dv
        #Sensitivity of error with respect to weights, use to change weights and move down the loss function
        de_dw = self.de_dy * dy_dw
        self.weights -= de_dw * self.learning_rate

        #Add in regularization at this layer
        #For each regulizer in the layer's list of regularizers
        for regulizer in self.regularizers:
            #Regulate the weights and prevent these moving too far so as to avoid overfitting i.e. the issue where the model fits predicitons to 'noise' as well as the signal.
            self.weights = regulizer.update(self)

        #Self.weights.transpose() = dy/dx - use the chain rule to find de_dx from de_dy, dy_dv and dv_dx
        self.de_dx = (self.de_dy * dy_dv) @ dv_dx
        #Where nodes are dropped out, remove these by setting the derivative of the error wrt these nodes to zero
        #Set de_dx to zero for each node where the i_dropout array is set to True
        de_dx_no_bias = self.de_dx[:, :-1]
        de_dx_no_bias[:, self.i_dropout] = 0

        #Remove the bias node from the gradient vector
        self.previous_layer.de_dy += de_dx_no_bias

        #Return derivatives at all layer expect the bias layer (the -1 excludes this)
        #return de_dx[:, :-1]

#Create the dense layer which inherits from GenericLayer
#Transform the input and output values so they tends to fall between -0.5 and 0.5
class RangeNormalization(GenericLayer):
    def __init__(self, training_data, previous_layer=None):
        self.previous_layer = previous_layer

        #Estimate the range based on a selection of the training data
        n_range_test = 100
        self.range_min = 1e10
        self.range_max = -1e10
        for _ in range(n_range_test):
            sample = next(training_data())
            if self.range_min > np.min(sample):
                self.range_min = np.min(sample)
            if self.range_max < np.max(sample):
                self.range_max = np.max(sample)
        self.scale_factor = self.range_max - self.range_min
        self.offset_factor = self.range_min
        self.size = sample.size
        self.reset()

    def forward_pass(self, **kwargs):
        if self.previous_layer is not None:
            self.x += self.previous_layer.y
        self.y = (self.x - self.offset_factor) / self.scale_factor - 0.5

    def backward_pass(self):
        #dy_dx = 1 / scale factor - ??
        #Hence, de_dx = de_dy * dy_dx
        self.de_dx = self.de_dy / self.scale_factor
        if self.previous_layer is not None:
            self.previous_layer.de_dy += self.de_dx

    def denormalize(self, transformed_values):
        #In case we need to reverse the normalization process
        min_val = self.expected_range[0]
        max_val = self.expected_range[1]
        scale_factor = 2 / (max_val - min_val)
        offset_factor = min_val - 1
        return transformed_values / scale_factor - offset_factor

class Difference(GenericLayer):
    """
    A difference layer calclates the difference between the Outputs of two
    earlier layers, the previous layer and another one from earlier back, Which
    we will call subtract_me_layer. The output of this layer is:
    y = previous_layer.y - subtract_me_layer.y
    """
    def __init__(self, previous_layer, subtract_me_layer):
        #Take information on the previous and subtract_me layers
        self.previous_layer = previous_layer
        self.subtract_me_layer = subtract_me_layer

        #Force the two layers inputted to have the same size to ensure operations work
        assert self.subtract_me_layer.y.size == self.previous_layer.y.size
        #Set size of this difference layer to the same as the two layers
        self.size = self.previous_layer.y.size

    def forward_pass(self, **kwargs):
        self.y = self.previous_layer.y - self.subtract_me_layer.y

    def backward_pass(self):
        self.previous_layer.de_dy += self.de_dy
        self.subtract_me_layer.de_dy -= self.de_dy
