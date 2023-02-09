from layer import Layer
import numpy as np


class Softmax(Layer):
    def __init__(self, f):
        self.cache = None
        self.f = f

    def forward(self, a, train = True):
        z_temp = np.exp(a-np.max(a, axis=1, keepdims=True))
        # z = z_temp / np.sum(z_temp, axis=1, keepdims=True)
        z = np.exp(a) / np.sum(np.exp(a), axis=1, keepdims=True)

        self.f.write("Softmax: \n")
        self.f.write("a: " + str(a) + "\n\n")
        self.f.write("z: " + str(z) + "\n\n")

        return z
        
    def backward(self, dz):
        da = dz
        return da
