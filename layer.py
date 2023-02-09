

class Layer:
    def __init__(self):
        raise NotImplementedError
    def forward(self, x, train):
        raise NotImplementedError
    def backward(self, dz):
        raise NotImplementedError

    def update(self, learning_rate):
        pass