from layer import Layer
import numpy as np

class BatchNormalization(Layer):
    def __init__(self):
        self.gamma = None
        self.beta = None
        self.cache = None
        self.d_gamma = None
        self.d_beta = None



    def init_params(self, input_shape):
        self.gamma = np.ones(input_shape)
        self.beta = np.zeros(input_shape)

    def forward(self, a, train = True):
        mean =  np.mean(a, axis=0)
        var = np.var(a, axis=0)
        z = (a - mean) / np.sqrt(var + 1e-8)

        if self.gamma is None:
            self.init_params(z.shape)


        if train == False:
            self.gamma = np.ones(z.shape)
            self.beta = np.zeros(z.shape)

        if self.gamma.shape != z.shape:
            self.gamma = np.ones(z.shape)
            self.beta = np.zeros(z.shape)
            
        z = self.gamma * z + self.beta
        self.cache = z

        return z


    def backward(self, dz):
        da = dz
        self.d_gamma = np.sum(dz * self.cache, axis=0)
        self.d_beta = np.sum(dz, axis=0)
        return da

    def update(self, learning_rate):
        self.gamma -= learning_rate * self.d_gamma
        self.beta -= learning_rate * self.d_beta
    

    