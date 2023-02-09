from layer import Layer
import numpy as np


class SigmoidActivation(Layer):
    def __init__(self):
        self.cache = None

    def forward(self, a, train = True):
        z = 1 / (1 + np.exp(-a))
        self.cache = z
        return z

    def backward(self, dz):
        z = self.cache
        da = dz * (1 - z) * z
        return da