import numpy as np
from input_layer import *
from hidden_layer import *
from output_layer import *
np.random.seed(seed = 1)

class RNN():
    def __init__(self,num_inputs,num_outputs,hidden_layers,non_linearity):
        self.input_layer = Input_Layer(num_inputs);
        self.hidden_layers = {}
        prev = self.input_layer.size
        for key,value in hidden_layers.iteritems():
            self.hidden_layers[key] = Hidden_Layer(non_linearity,value,prev,self.input_layer.size)
            prev = self.hidden_layers[key].neurons

        self.output_layer = Output_Layer(num_outputs,prev,non_linearity)



    def forward_pass(self,inputs):
        current_output = self.input_layer.compute(inputs)
        for index,h_layer in self.hidden_layers.iteritems():
            current_output = h_layer.compute(current_output)
        current_output = self.output_layer.compute(current_output)
        return current_output

    def train(self,inputs,targets):
        for i in range(10000):
