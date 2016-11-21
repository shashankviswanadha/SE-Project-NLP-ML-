import numpy as np
from ANN import *
from hidden_layer import *
from input_layer import *
from output_layer import *
from ActivationFunction import *
data = np.loadtxt("Data.txt")
import matplotlib.pyplot as plt

hid = {1:10}
t0 = data[:,0]
#t1 = data[:,1]
#t2 = data[:,2]
#t3 = data[:,3]
te = data[:,1]
#te[0] = te[0] - 3
ma = mi = te[0]
maa = mii = t0[0]
for x in range(1,len(te)):
    #te[x] = te[x] - 3
    if te[x] > ma:
        ma = te[x]
    if te[x] < mi:
        mi = te[x]
ra = ma - mi
for x in range(len(te)):
    te[x] = (te[x]/ra) - (mi/ra)

for x in range(1,len(t0)):
    if t0[x] > maa:
        maa = t0[x]
    if t0[x] < mii:
        mii = t0[x]
rat = maa - mii
for x in range(0,len(t0)):
    t0[x] = (t0[x] - mii)/rat
inpp = np.array([t0]).T
tarr = np.array([te]).T
inp = inpp
tar = tarr
np.savetxt('inp.txt',tar)
af = sigmoid
plt.plot(inp,tar,'ro')
plt.show()
#num_inputs,num_outputs,hidden_layers,non_linearity,learning_rate
temp1 = np.array([[0,0,0],
                  [0,0,1],
                  [0,1,0],
                  [0,1,1],
                  [1,0,0],
                  [1,0,1],
                  [1,1,0],
                  [1,1,1]])
temp2 = np.array([[0,0,0,0,0,1,0],[0,1,1,0,0,1,1],[0,1,1,0,0,0,0],[0,1,0,1,0,0,1],[0,1,1,1,1,0,1],[0,1,0,0,1,0,0],[0,1,0,0,1,1,0],[1,1,1,1,1,1,1]])
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1],
                [0,0,0]])

# output dataset
y = np.array([[0,1,1,0,1]]).T

ann = ANN(1, 1, hid, af, 0.001, 10000)
out = ann.train(inp,tar)
#print out
"""for x in range(len(out)):
    if out[x][0] > 0.5:
        out[x][0] = 1
    else:
        out[x][0] = 0"""
'''for k in range(len(tar)):
    tar[k] = mii + out[k]*rat'''
print out
np.savetxt('test.txt', out)
print "avg error = ",ann.output_layer.mse,ann.output_layer.avg_er
plt.plot(inp,tar,'ro')
plt.plot(inp,out,'ro')
plt.show()
"""C = te - out
np.savetxt('test.txt', C)
A = np.empty_like(C)
for i in range(len(C)):
    A[i][0] = i
plt.plot(A,C,'ro')
plt.show()"""
"""sum_error = 0
for i in range(0,len(out)):
    for j in range(0,len(out[0])):
        sum_error += abs(out[i][j] - temp2[i][j])
print sum_error/56"""
