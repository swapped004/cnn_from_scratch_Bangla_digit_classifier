from layer import Layer
import numpy as np
import math


class ReluActivation(Layer):
    def __init__(self):
        self.cache = None

    def forward(self, a, train = True):
        self.cache = a
        return np.maximum(0, a)

    def backward(self, dz):
        a = self.cache
        return dz * (a > 0)



#test code here

if __name__ == "__main__":
    relu1 = ReluActivation()

    #make a random input of size shape (16,3,28,28)

    a = np.random.randn(16,5,28,28)

    #forward pass

    z = relu1.forward(a)

    print("z shape: ", z.shape)
    print("z: ", z)

    #backward pass
    dz = np.random.randn(16,5,28,28)

    da = relu1.backward(dz)

    #backward pass
    print("da shape: ", da.shape)
    print("da: ", da)