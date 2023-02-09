from layer import Layer
import numpy as np
import math


class MaxPooling(Layer):
    def __init__(self, kernel_size = 2, stride = 2, f = None):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None

        self.f = f


    def forward(self, a, train = True):
        batch_size, input_channel, input_height, input_width = a.shape
        output_height = (input_height - self.kernel_size) // self.stride + 1
        output_width = (input_width - self.kernel_size) // self.stride + 1

        self.cache = a

        z = np.zeros((batch_size, input_channel, output_height, output_width))


        for i in range(output_height):
            for j in range(output_width):
                z[:, :, i, j] = np.max(a[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size], axis=(2,3))


        assert(z.shape == (batch_size, input_channel, output_height, output_width))
        
        return z




    def backward(self, dz):
        a = self.cache

        da = np.zeros(a.shape)


        _,_,output_height, output_width = dz.shape

        for i in range(output_height):
            for j in range(output_width):
                window = a[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size]
                mask = (window == np.max(window, axis=(2,3), keepdims=True))
                da[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size] += mask * dz[:, :, i, j][:, :, None, None]


        assert(da.shape == a.shape)

        return da


    def print_params(self, f):
        pass




#test_maxpooling.py

if __name__ == '__main__':

    maxpool1 = MaxPooling()

    #make a random input of size shape (16,3,28,28)

    a = np.random.randn(2, 3, 4, 4)

    #forward pass

    z = maxpool1.forward(a)

    print("z shape: ", z.shape)

    f = open("maxpooling.txt", "w")
    f.write("a:\n"+str(a)+"\n")
    f.write("z:\n"+str(z)+"\n")	

    #backward pass
    dz = np.random.randn(2,3,2,2)

    da = maxpool1.backward(dz)

    print("da shape: ", da.shape)

    f.write("dz:\n"+str(dz)+"\n")
    f.write("da:\n"+str(da)+"\n")


    

