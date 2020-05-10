#Import data
import data_loader_nordic_runes as dat
#Import key components of the neural network i.e. activation functions, ANN class, layers, loss function, regulizers
import nn_framework.activation as activation
import nn_framework.framework as framework
from nn_framework.layer import Dense
import nn_framework.error_fun as error_fun
from nn_framework.regularization import L1, L2, Limit
#Import printer function for visualising neural networks
from autoencoder_viz import Printer

#Set up the hidden layers as a list. Each entry is a layer and the valueof the entry is the number of nodes in the list
N_NODES = [24]

#Retrieve the training and evaluation datasets from the data_loader script
training_set, evaluation_set = dat.get_data_sets()

sample = next(training_set())
input_value_range = (0, 1)
n_pixels = sample.shape[0] * sample.shape[1]
printer = Printer(input_shape=sample.shape)

#n_nodes acts as a list  of each of the layers of nodes. [n_pixels] is the input layer which contains the total number of pixels from the image data we import, then
#N_NODES sets the number of nodes in the hidden layer, currently set to 5. The output layer also corresponds with the total number and therefore arrangement of pixels
n_nodes = [n_pixels] + N_NODES + [n_pixels]
model = []
#For each layer of nodes in the model
for i_layer in range(len(n_nodes) - 1):
    #Add dense layers in the form of the Dense class. This is intialised by entering the number of inputs, ouputs and tanh activation function)
    #Why is there one extra output
    new_layer = Dense(
        n_nodes[i_layer],
        n_nodes[i_layer + 1],
        activation.tanh,
    )
    #Add regularizer to the layer - use add_regularizer method to add L1, L2 and custom objects
    #Why run both -  both have pros that work together. L1 frees up nodes that don't do anything for the model and speed up processing. L2 pulls down on really high weights
    new_layer.add_regularizer(L1())
    new_layer.add_regularizer(L2())
    new_layer.add_regularizer(Limit(1.0))
    #Append the new layer to the model
    model.append(new_layer)

#Run the autoencoder
#Initialise the ANN class with the model in terms of nodes, layers and activation function; the loss function; and the range for normalising the inputs
#Also, set up the printer function so we can visualse the Neural Network as it computes
autoencoder = framework.ANN(
    model=model,
    error_fun=error_fun.abs,
    printer=printer,
    expected_range=input_value_range,
)
autoencoder.train(training_set)
autoencoder.evaluate(evaluation_set)
