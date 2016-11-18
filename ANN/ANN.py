import numpy as np
from input_layer import *
from hidden_layer import *
from output_layer import *
np.random.seed(seed = 1)

class ANN():
    def __init__(self,num_inputs,num_outputs,hidden_layers,non_linearity,learning_rate):
        self.input_layer = Input_Layer(num_inputs);
        self.hidden_layers = {}
        self.learning_rate = learning_rate
        prev = self.input_layer.size
        for key,value in hidden_layers.iteritems():
            self.hidden_layers[key] = Hidden_Layer(non_linearity,value,prev,learning_rate)
            prev = self.hidden_layers[key].neurons
        self.num_hidden_layers = len(self.hidden_layers)
        self.output_layer = Output_Layer(num_outputs,prev,non_linearity,learning_rate)



    def forward_pass(self,inputs):
        current_output = self.input_layer.compute(inputs)
        for index,h_layer in self.hidden_layers.iteritems():
            current_output = h_layer.compute(current_output)
        current_output = self.output_layer.compute(current_output)
        return current_output

    def train(self,inputs,targets):
        for j in xrange(10000):
            self.forward_pass(inputs)
            curr_data = self.output_layer.backprop(targets)
            for i in range(self.num_hidden_layers,0,-1):
                curr_data = self.hidden_layers[i].backprop(curr_data)

        temp = self.forward_pass(inputs)
        return temp
