import numpy as np
from ANN import *
from hidden_layer import *
from input_layer import *
from output_layer import *
from ActivationFunction import *
data = np.loadtxt("rmsout_28H_5000_RMSCost.txt")
import matplotlib.pyplot as plt
import random

hid = {1:20}
t0 = data[:,0]
t1 = data[:,1]
t2 = data[:,2]
#t3 = data[:,3]
te = data[:,4]
#te[0] = te[0] - 3
ma = mi = te[0]
#maa = mii = t0[0]
for x in range(1,len(te)):
    #te[x] = te[x] - 3
    if te[x] > ma:
        ma = te[x]
    if te[x] < mi:
        mi = te[x]
ra = ma - mi
for x in range(len(te)):
    te[x] = (te[x]/ra) - (mi/ra)
inp0_test = []
inp1_test = []
inp2_test = []
inp0_train = []
inp1_train = []
inp2_train = []
out_test = []
out_train = []
for x in range(len(te)):
    if(x%5 == 0):
        inp0_test.append(t0[x])
        inp1_test.append(t1[x])
        inp2_test.append(t2[x])
        out_test.append(te[x])
    else:
        inp0_train.append(t0[x])
        inp1_train.append(t1[x])
        inp2_train.append(t2[x])
        out_train.append(te[x])

'''for x in range(1,len(t0)):
    if t0[x] > maa:
        maa = t0[x]
    if t0[x] < mii:
        mii = t0[x]
rat = maa - mii
for x in range(0,len(t0)):
    t0[x] = (t0[x] - mii)/rat'''
'''for o in range(len(out_train)):
    out_train[o] = out_train[o] + random.uniform(0.0,0.05)'''

inpp = np.array([inp0_train,inp1_train,inp2_train]).T
tarr = np.array([out_train]).T
inp = inpp
tar = tarr
inpp = np.array([inp0_test,inp1_test,inp2_test]).T
tarr = np.array([out_test]).T
inp_test = inpp
out_te = tarr
np.savetxt('input_training.txt',inp)
np.savetxt('output_training.txt',tar)
np.savetxt('output_testing.txt',out_te)
af = sigmoid
#plt.plot(inp_test,out_te,'ro')
#plt.show()
#num_inputs,num_outputs,hidden_layers,non_linearity,learning_rate
#plt.plot(inp,tar,'ro',)
#plt.show()
ann = ANN(3, 1, hid, af, 0.001, 5000)
t = ann.train(inp,tar,inp_test,out_te)
#print out
#print "ms error = ",ann.output_layer.mse,"avg error = ",ann.output_layer.avg_er
"""for x in range(len(out)):
    if out[x][0] > 0.5:
        out[x][0] = 1
    else:
        out[x][0] = 0"""
'''for k in range(len(tar)):
    tar[k] = mii + out[k]*rat'''
out = ann.forward_pass(inp_test)
np.savetxt('obtained_output_training.txt', t)
np.savetxt('obtained_output_testing.txt', out)

add = 0
tr = 0
for x in range(len(out)):
    add += abs(out[x][0] - out_te[x][0])
for x in range(len(t)):
    tr += abs(t[x][0] - tar[x][0])
print 'train error : ',tr,tr/len(t)
print 'test error : ',add,add/len(out)
plt.plot(out_te,out,'ro',c='black')
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
