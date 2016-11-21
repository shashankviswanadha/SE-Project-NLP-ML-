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
        self.avg_er = None
        self.mse = None
    def initialize_weights(self):
        self.W = 2*np.random.random((self.num_prev_layer,self.neurons)) - 1

    def compute(self,inputs):
        self.inputs = inputs
        self.outputs = self.nonlinearity(np.dot(inputs,self.W))
        return self.outputs

    def backprop(self,targets):
        self.error =  targets - self.outputs#0.5*np.square(targets - self.outputs)#((targets - self.outputs) ** 2).mean(axis=1) targets - self.outputs
        summ = 0
        summ2 = 0
        self.delta = self.error * self.nonlinearity(self.outputs, deriv = True)
        self.W += self.inputs.T.dot(self.delta) * self.learning_rate
        for i in range(len(targets)):
            for j in range(len(targets[0])):
                summ2 += (self.outputs[i][j] - targets[i][j])**2
                summ += abs(self.outputs[i][j] - targets[i][j])
        self.avg_er = summ/(len(targets) * len(targets[0]))
        self.mse = summ2/(2*len(targets) * len(targets[0]))
        return (self.delta,self.W)
