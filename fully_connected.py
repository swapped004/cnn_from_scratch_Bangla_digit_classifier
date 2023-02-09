from layer import Layer
import numpy as np


class FullyConnected(Layer):
    def __init__(self, output_size, f = None):
        self.output_size = output_size
        self.W = None
        self.b = None
        self.cache = None
        self.dW = None
        self.db = None

        self.f = f

    def init_weights(self, input_size):
        self.W = np.random.randn(input_size, self.output_size) * np.sqrt(2.0 / input_size)
        self.b = np.zeros(self.output_size)


    def forward(self, a, train = True):
        if self.W is None:
            self.init_weights(a.shape[1])

        self.cache = a
        z = np.dot(a, self.W) + self.b

        assert(z.shape == (a.shape[0], self.output_size))

        self.print_params(self.f)

        return z

    def backward(self, dz):
        a = self.cache

        self.dW = np.dot(a.T, dz)
        self.db = np.sum(dz, axis=0)
        da = np.dot(dz, self.W.T)

        assert(da.shape == a.shape)
        assert(self.dW.shape == self.W.shape)
        assert(self.db.shape == self.b.shape)

        self.print_params(self.f)

        return da

    def update(self, learning_rate):
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db


    def print_params(self,f):
        f.write("Fully Connected Layer\n")
        f.write("W: " + str(self.W) + "\n\n")
        f.write("dW: " + str(self.dW) + "\n\n\n")

        f.write("b: " + str(self.b) + "\n\n")
        f.write("db: " + str(self.db) + "\n\n")


#test code
if __name__ == "__main__":
    fc1 = FullyConnected(10)

    #make a random input of size shape ((16, 100)
    a = np.random.randn(16, 100)

    #forward pass
    z = fc1.forward(a)

    print("z shape: ", z.shape)

    #backward pass
    #make a random dz of the same shape as z
    dz = np.random.randn(16, 10)

    da = fc1.backward(dz)

    print("da shape: ", da.shape)

    f = open("fully_connected.txt", "w")

    fc1.print_params(f)

    fc1.update(0.01)

    fc1.print_params(f)

    