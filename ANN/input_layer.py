import numpy as np
np.random.seed(seed=1)

class Input_Layer():
    def __init__(self,num_inputs):
        self.values = None
        self.size = num_inputs
        self.output = None
    def compute(self,inputs):
        self.values = inputs
        self.output = self.values
        return self.output
