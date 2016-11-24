import numpy as np
from input_layer import *
from hidden_layer import *
from output_layer import *
np.random.seed(seed = 1)
import matplotlib.pyplot as plt
class ANN():
    def __init__(self,num_inputs,num_outputs,hidden_layers,non_linearity,learning_rate,iterations):
        self.input_layer = Input_Layer(num_inputs);
        self.hidden_layers = {}
        self.learning_rate = learning_rate
        self.iterations = iterations
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

    def train(self,inputs,targets,test_inputs,test_targets):
        epochs = []
        train_error = []
        test_error = []
        for j in xrange(self.iterations):
            self.forward_pass(inputs)
            curr_data = self.output_layer.backprop(targets)
            for i in range(self.num_hidden_layers,0,-1):
                curr_data = self.hidden_layers[i].backprop(curr_data)
            test_outputs = self.forward_pass(test_inputs)
            summ = summ2 = 0
            for i in range(len(test_targets)):
                for k in range(len(test_targets[0])):
                    summ2 += (test_outputs[i][k] - test_targets[i][k])**2
                    summ += abs(test_outputs[i][k] - test_targets[i][k])
            avg_er = summ/(len(test_targets) * len(test_targets[0]))
            mse = summ2/(2*len(test_targets) * len(test_targets[0]))
            epochs.append(j)
            train_error.append(self.output_layer.mse)
            test_error.append(mse)

        plt.plot(epochs,test_error,c = 'green',label = 'testing error')
        plt.plot(epochs,train_error,c='red',label = 'training error')
        plt.xlabel('epochs')
        plt.ylabel('error')
        plt.legend()
        plt.show()
        print '----------------------------'
        print 'tr er mse, avg : ',self.output_layer.mse,self.output_layer.avg_er
        print 'te re mse , avg: ',summ,mse,avg_er
        temp = self.forward_pass(inputs)
        return temp
