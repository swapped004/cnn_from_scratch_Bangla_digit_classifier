import numpy as np
from tqdm import tqdm

from layer import Layer

from convolution import Convolution
from relu_activation import ReluActivation
from maxpooling import MaxPooling
from flattening import Flattening
from fully_connected import FullyConnected
from softmax import Softmax

from sigmoid_activation import SigmoidActivation
from batch_normalization import BatchNormalization


from data_loader import DataLoader



class CNN:
    def __init__(self,learning_rate=0.001):
        self.learning_rate = learning_rate
        self.layers = []
    

    def add(self, layer):
        self.layers.append(layer)


    def init_architecture(self):
        self.add(Convolution(3, 3, 5, 1, 1))
        self.add(ReluActivation())
        self.add(MaxPooling(2, 2))
        self.add(Flattening())
        self.add(FullyConnected(10))
        self.add(Softmax())



    def forward(self, x, train):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dz):
        for layer in reversed(self.layers):
            dz = layer.backward(dz)
            layer.update(self.learning_rate)
        return dz

    def train(self, mini_batches, epochs=100):
        for epoch in tqdm(range(epochs)):
            for mini_batch in tqdm(mini_batches):
                x, y = mini_batch
                z = self.forward(x, train=True)
                dz = z - y
                self.backward(dz)

        print("Training complete !!!")


    def test_labels(self, x):
        z = self.forward(x, train=False)
        return np.argmax(z, axis=1)



#test code
if __name__ == "__main__":
    cnn = CNN(learning_rate=0.001)
    cnn.init_architecture()

    #load data
    data_loader = DataLoader(grayscale=False)
    X_train, y_train = data_loader.load_data("datasets/training-b.csv")

    mini_batches = data_loader.mini_batches(X_train, y_train, batch_size=32)

    cnn.train(mini_batches, epochs=100)


    X_test = data_loader.load_test_data("datasets/testing-b")

    labels = cnn.test_labels(X_test)

    print(labels)





        

    