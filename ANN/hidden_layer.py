import numpy as np
np.random.seed(seed=1)


class Hidden_Layer():
    def __init__(self,nonlinearity,num_neurons,num_prev_layer,learning_rate):
        self.nonlinearity  = nonlinearity
        self.neurons = num_neurons
        self.num_prev_layer = num_prev_layer
        self.W = None
        self.learning_rate = learning_rate
        self.initialize_weights()
        self.outputs = None
        self.inputs = None
        self.error = None
        self.delta = None

    def initialize_weights(self):
        self.W = 2*np.random.random((self.num_prev_layer,self.neurons)) - 1

    def compute(self,inputs):
        self.inputs = inputs
        self.outputs = self.nonlinearity(np.dot(inputs,self.W))
        return self.outputs

    def backprop(self,next_data):
        next_delta = next_data[0]
        next_W = next_data[1]
        #print self.W
        self.error = next_delta.dot(next_W.T)
        self.delta = self.error * self.nonlinearity(self.outputs, deriv = True)
        self.W += self.inputs.T.dot(self.delta) * self.learning_rate
        return (self.delta,self.W)
