import numpy as np
np.random.seed(seed=1)

class Output_Layer():
    def __init__(self,num_outputs,num_prev_layer,nonlinearity):
        self.neurons = num_outputs
        self.num_prev_layer = num_prev_layer
        self.W = None
        self.nonlinearity = nonlinearity
        self.outputs = None
        self.inputs = None
        self.initialize_weights()

    def initialize_weights(self):
        self.W = 2*np.random.random((self.num_prev_layer,self.neurons)) - 1

    def compute(self,inputs):
        self.inputs = inputs
        self.outputs = self.nonlinearity(np.dot(inputs,self.W))
        return self.outputs
