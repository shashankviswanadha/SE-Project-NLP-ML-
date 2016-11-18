import numpy as np
np.random.seed(seed=1)

class Output_Layer():
    def __init__(self,num_outputs,num_prev_layer,nonlinearity,learning_rate):
        self.neurons = num_outputs
        self.num_prev_layer = num_prev_layer
        self.W = None
        self.learning_rate = learning_rate
        self.nonlinearity = nonlinearity
        self.outputs = None
        self.inputs = None
        self.error = None
        self.delta = None
        self.initialize_weights()

    def initialize_weights(self):
        self.W = 2*np.random.random((self.num_prev_layer,self.neurons)) - 1

    def compute(self,inputs):
        self.inputs = inputs
        self.outputs = self.nonlinearity(np.dot(inputs,self.W))
        return self.outputs

    def backprop(self,targets):
        self.error = targets - self.outputs#0.5*np.square(targets - self.outputs)#((targets - self.outputs) ** 2).mean(axis=1)
        #summ = 0
        self.delta = self.error * self.nonlinearity(self.outputs, deriv = True)
        self.W += self.inputs.T.dot(self.delta) * self.learning_rate
        """for i in range(10000):
            summ += abs(self.outputs[i][0] - targets[i][0])
        print summ/10000"""
        return (self.delta,self.W)
