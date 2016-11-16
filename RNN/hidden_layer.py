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
        self.outputs = None
        self.inputs = None

    def initialize_weights(self):
        self.W_rec = 2*np.random.random((self.num_prev_layer,self.neurons)) - 1
        self.W = 2*np.random.random((self.num_prev_layer,self.neurons)) - 1

    def initialize_memory(self,num_inputs):
        self.memory.append(np.zeros(self.neurons))


    def compute(self,inputs):
        self.inputs = inputs
        self.outputs = self.nonlinearity(np.dot(inputs,self.W) + np.dot(self.memory[-1],self.W_rec))
        self.memory.append(self.outputs)
        return self.outputs
