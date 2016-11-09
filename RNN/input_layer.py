import numpy as np
np.random.seed(seed=1)

class Input_Layer():
    def __init__(self,values):
        self.values = values
        self.size = values.shape()
