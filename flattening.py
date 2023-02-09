from layer import Layer
import numpy as np
import math

class Flattening(Layer):
    def __init__(self):
        self.cache = None

    def forward(self, a, train = True):
        self.cache = a.shape
        z = a.reshape(a.shape[0], -1)
        return z

    def backward(self, dz):
        input_shape = self.cache
        da = dz.reshape(input_shape)
        return da