import numpy as np
from ANN import *
from hidden_layer import *
from input_layer import *
from output_layer import *
from ActivationFunction import *
data = np.loadtxt("Data.txt")
"""hid = {1:4}
inp = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
tar = np.array([[0],
			[1],
			[1],
			[0]])
af = sigmoid
#self,num_outputs,hidden_layers,non_linearity
ann = ANN(3, 1, hid, af, 50)
out = ann.backprop(inp,tar)
print "output : ",out"""

hid = {}
t = data[:,0]
te = data[:,1]
ma = mi = te[0]
maa = mii = t[0]
for x in range(1,len(te)):
    if te[x] > ma:
        ma = te[x]
    if te[x] < mi:
        mi = te[x]
ra = ma - mi
for x in range(0,len(te)):
    te[x] = (te[x] - mi)/ra

for x in range(1,len(t)):
    if t[x] > maa:
        maa = t[x]
    if t[x] < mii:
        mii = t[x]
rat = ma - mi
for x in range(0,len(te)):
    t[x] = (t[x] - mii)/rat
inp = np.array([t]).T
tar = np.array([te]).T
np.savetxt('inp.txt',tar)
af = sigmoid
#num_inputs,num_outputs,hidden_layers,non_linearity,learning_rate
ann = ANN(1, 1, hid, af, 1)
out = ann.train(inp,tar)
np.savetxt('test.txt', out)
sum_error = 0
for i in range(0,10000):
    sum_error += abs(out[i][0] - tar[i][0])
print sum_error/10000
