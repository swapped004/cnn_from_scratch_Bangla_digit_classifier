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
        self.conv1 = open("convolution1.txt", "w")
        self.add(Convolution(3, 3, 5, 1, 1, self.conv1))
        
        self.relu1 = open("relu1.txt", "w")
        self.add(ReluActivation())
        
        self.maxpool1 = open("maxpool1.txt", "w")
        self.add(MaxPooling(2, 2, self.maxpool1))
       
        
        # self.conv2 = open("convolution2.txt", "w")
        # self.add(Convolution(5, 3, 5, 1, 1, self.conv2))
        
        # self.relu2 = open("relu2.txt", "w")
        # self.add(ReluActivation())

        # self.maxpool2 = open("maxpool2.txt", "w")
        # self.add(MaxPooling(2, 2, self.maxpool2))
        
        self.flatten = open("flatten.txt", "w")
        self.add(Flattening())

        self.fc1 = open("fullyconnected1.txt", "w")
        self.add(FullyConnected(10, self.fc1))

        self.bn1 = open("batchnorm1.txt", "w")
        self.add(BatchNormalization())

        # self.add(ReluActivation())
        

        # self.fc2 = open("fullyconnected2.txt", "w")
        # self.add(FullyConnected(10, self.fc2))

        # self.bn2 = open("batchnorm2.txt", "w")
        # self.add(BatchNormalization())

        self.softmax = open("softmax.txt", "w")
        self.add(Softmax(self.softmax))



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


    def test_accuracy(self, x, y):
        z = self.forward(x, train=False)

        print("z shape: ", z.shape)
        print("z: ", z)

        y_pred = np.argmax(z, axis=1)
        y_true = np.argmax(y, axis=1)

        #print predictions and true labels
        print("Predictions: ", y_pred)
        print("True labels: ", y_true)

        accuray = np.sum(y_pred == y_true) / len(y_true) * 100
        print("Accuracy: ", accuray, "%")


#test code
if __name__ == "__main__":
    #change seed for different results
    np.random.seed(0)


    cnn = CNN(learning_rate=0.001)
    cnn.init_architecture()

    #load data
    data_loader = DataLoader(grayscale=False)
    X_train, y_train, X_test, y_test = data_loader.load_data("datasets/training-a.csv", test_size = 0.5)

    mini_batches = data_loader.mini_batches(X_train, y_train, batch_size=64)

    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)
    print("X_test shape: ", X_test.shape)
    print("y_test shape: ", y_test.shape)


    cnn.train(mini_batches, epochs=100)

    # X_test, y_test = data_loader.load_data("datasets/training-b.csv")

    cnn.test_accuracy(X_test, y_test)


    # X_test = data_loader.load_test_data("datasets/testing-b")

    # labels = cnn.test_labels(X_test)

    # print(labels)





        

    