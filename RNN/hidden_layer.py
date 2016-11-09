import numpy as np
np.random.seed(seed=1)


class Hidden_Layer():
    def __init__(self,nonlinearity,num_neurons,num_prev_layer,num_inputs):
        self.nonlinearity  = nonlinearity
        self.neurons = num_neurons
        self.num_prev_layer = num_prev_layer
        self.W = None
        self.W_rec = None
        self.memory = list()
        self.initialize_weights()
        self.initialize_memory(num_inputs)


    def initialize_weights(self):
        self.W_rec = 2*np.random.random((self.num_prev_layer,self.neurons)) - 1
        self.W = 2*np.random.random((self.num_prev_layer,self.neurons)) - 1

    def initialize_memory(self,num_inputs):
        self.memory.append(np.zeros(num_inputs,self.neurons))
