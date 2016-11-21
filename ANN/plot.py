import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt("Sine_data.txt")
inp = data[:,0]
out = data[:,1]
plt.plot(inp,out,'ro')
plt.show()
