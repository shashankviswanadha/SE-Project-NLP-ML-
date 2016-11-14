# -*- coding: utf-8 -*-
"""
Created on Mon Oct 03 20:48:21 2016

@author: kaust
"""
#importing necessary modules
import math
import random
import numpy as np

np.seterr(all = 'ignore')   #To ignore floating point errors. No exception is raised

#Activation Function
#Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
#Derivative of sigmoid function
#derivative = sigmoid(y) * (1- sigmoid(y)) 

def diffsigmoid(y):
 
    return y * (1.0 - y)

#Tanh for activation function

def tanh(x):
    return np.tanh(x)
#Derivative of tanh sigmoid

def difftanhx(y):
    return 1 - y*y
    

class FF_NeuralNetwork(object):
    """
    This basic feedforward neural network has three layers; input, hidden 
    and output layers. Number of hidden layers is user defined when we initialize the neural net. 
    
    """
    
    def __init__(self, input_layer, hidden_layer, output_layer, iterations, learning_rate, momentum, anneal_rate):
        """
        input_layer: no. of input neurons
        hidden_layer: no. of hidden layers
        output_layer: no. of output neurons
        """
        
        #Initializing the params
        
        self.input_layer = input_layer #1 added for bias node
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        
        self.iterations = iterations
        self.learning_rate = learning_rate        
        self.momentum = momentum
        self.anneal_rate = anneal_rate
        
        #Array initialization for activation
        
        self.act_input = [1.0] * self.input_layer
        self.act_hidden = [1.0] * self.hidden_layer
        self.act_output = [1.0] * self.output_layer
        
        #Arrays initialization for random weights
        input_size = 1.0/self.input_layer**(0.5)
        output_size = 1.0/self.hidden_layer**(0.5)
        
        self.wt_input = np.random.normal(loc = 0, scale = input_size, size = (self.input_layer,self.hidden_layer))
        
        self.wt_output = np.random.normal(loc = 0, scale = output_size, size = (self.hidden_layer, self.output_layer))
        
        #Array initialzation for storing the changes.
        #Updated after every iteration based on the extent to which the weightsneed to be changed for the following iteration
        
        self.change_input = np.zeros((self.input_layer, self.hidden_layer))
        self.change_output = np.zeros((self.hidden_layer, self.output_layer))
        
        
    def feed_forward(self, inputs):
            # param inputs: input data
            # return value: output vector ( activated)
        
            
           
           #Activation of the inputs
            for i in xrange(self.input_layer):
                #To ignore the bias node
              #  print inputs[i]
                self.act_input[i] = inputs[i]
            #Activation for the hidden layers
            for j in xrange(self.hidden_layer):
                sum1 = 0.0
                
                for k in xrange(self.input_layer):
                    sum1 += self.act_input[k] * self.wt_input[k][j]
                #self.act_hidden[j] = sigmoid(sum1)
                self.act_hidden[j] = sigmoid(sum1)
                
           # print ("act_hidden",self.act_hidden)
            #print self.output_layer
           #Output activation
            for p in xrange(self.output_layer):
                 sum2 = 0.0
                 for q in xrange(self.hidden_layer):
                     sum2 += self.act_hidden[q] * self.wt_output[q][p]
                 self.act_output[p] = sigmoid(sum2)
            print self.act_input[:],len(self.act_input)
            print '*********************'
            print self.act_hidden[:], len(self.act_hidden)
            print '*********************'
            print self.act_output[:], len(self.act_output)
            return self.act_output[:]
    
    def back_propogate(self, targets):
                       
        # param targets: y values expected
        #return adjusted weights
        #if len(targets) != self.output_layer:
           # raise ValueError("Length of target vector is not proper")
       #The calculation of delta i.e, gradient gives you the direction in which the weights are to be updated to arrive fastest at a local minima
       #Error term calculation for outputs
        
       
       
        delta_output = [0.0] *self.output_layer
        #Error calculation
        for i in xrange(self.output_layer):
            error = 0.5 * (targets[i] - self.act_output[i]) ** 2
            delta_output[i] = diffsigmoid(self.act_output[i]) * error
            print self.act_output[i]
        #Error term calculation for hidden layers
        delta_hidden = [0.0] * self.hidden_layer
        for j in xrange(self.hidden_layer):
            error = 0.0
            for k in xrange(self.output_layer):
                error += delta_output[k] * self.wt_output[j][k]
            delta_hidden[j] = diffsigmoid(self.act_hidden[i])*error
            
        #Weight update; hidden -> output
        for p in xrange(self.hidden_layer):
            for q in xrange(self.output_layer):
                change = delta_output[q] *self.act_hidden[p]
                self.wt_output[p][q] -= self.learning_rate * change + self.change_output[p][q] * self.momentum
                self.change_output[p][q] = change


        #Weight update; input -> hidden
        for r in xrange(self.input_layer):
            for s in xrange(self.hidden_layer):
                change = delta_hidden[s] * self.act_input[r]
                self.wt_input[r][s] -= self.learning_rate * change + self.change_input[r][s] * self.momentum
                self.change_input[r][s] = change
                
        return error     
    
    def test_NN(self, data):
        
        for k in data:
            print(k[1], '--', self.feed_forward(k[0]))
    def train_NN(self, data):
        inputs1, targets = [] , []
        for i in xrange(self.iterations):
            error  = 0.0
           
            random.shuffle(data)
            #print data
            for j in data:
                
                inputs1.append(j[0])
                
                #print j[0]
                #print len(inputs)
                targets.append( j[1])
                
                
                self.feed_forward(inputs1)
                   
                error += self.back_propogate(targets)
        
            
            with open('error.txt', 'a') as errorf_write:
                errorf_write.write(str(error)+'\n')
                errorf_write.close()
                
            self.learning_rate = self.learning_rate * (self.learning_rate / (self.learning_rate + (self.learning_rate*self.anneal_rate)))
       
               
        
           
        print inputs1    #rint inputs
    def predict(self, X):
        predicts = []
        
        for i in X:
            predicts.append(self.feed_forward(i))
        return predicts
        
        
        
def demo():    
    
    def load_data():
        
       
    
        data = np.loadtxt('Data.txt')
        
        y = data[:,1]
        data = data[:,0]
        
        out = []
        
        #print data, y
        for i in xrange(data.shape[0]):
            ff = list((data[i].tolist(), y[i].tolist()))
            out.append(ff)
            
        
        return out


    X = load_data()
  
#    targets1 = []
#    inputs1 = []
#    for u in X:
#        inputs1.append(u[0])
#        targets1.append(u[1])
#        #print u
#    #print "Inps --",inputs1]
  
#    #print targets1
    print X[3]
    NN = FF_NeuralNetwork(1, 100, 10, iterations = 50, learning_rate = 0.5, momentum = 0.5, anneal_rate = 0.01 )
   # print NN.input_layer, NN.hidden_layer
    
        
    NN.train_NN(X)
    NN.test_NN(X)


if __name__ == '__main__':
    
    demo()  
                
        
             
        
        
        
        
    
    
        