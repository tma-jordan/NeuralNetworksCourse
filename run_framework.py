import data_loader_two_by_two as dat
import nn_framework.activation as activation
import nn_framework.framework as framework
import nn_framework.layer as layer

N_NODES = [7, 4, 6]

training_set, evaluation_set = dat.get_data_sets()

sample = next(training_set())
input_value_range = (0, 1)
n_pixels = sample.shape[0] * sample.shape[1]


#n_nodes acts as a list or vector of each of the layers of nodes. [n_pixels] is the input layer which contains the total number of pixels from the image data we import, then
#N_NODES sets the number of nodes in the hidden layer, currently set to 5. The output layer is currently also set to contain [n_pixels]
n_nodes = [n_pixels] + N_NODES + [n_pixels]
model = []
#For each layer of nodes we iterate through...
for i_layer in range(len(n_nodes) - 1):
    #Add dense layers in the form of the Dense class. This is intialised by entering the number of inputs, ouputs and tanh activation function)
    #Why is there one extra output
    model.append(layer.Dense(
        n_nodes[i_layer],
        n_nodes[i_layer + 1],
        activation.tanh
    ))

#Run the autoencoder
autoencoder = framework.ANN(
    model=model,
    expected_range=input_value_range,
)
autoencoder.train(training_set)
