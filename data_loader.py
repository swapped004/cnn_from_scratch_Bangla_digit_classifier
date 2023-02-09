import numpy as np
import pandas as pd
import cv2
import math
import os


class DataLoader:
    def __init__(self, grayscale = True):
        self.grayscale = grayscale

    def load_test_data(self, path):
        #for all images in the test folder

        #read the image files and store them in X_train

        X_test = []

        for file_name in os.listdir(path):
            img_name = path+"/"+file_name

            #check if the file is an image
            if not img_name.endswith(".png"):
                continue

            # print(img_name)
            
            img = cv2.imread(img_name)

            #resize the image to (n,n)
            img = cv2.resize(img, (32,32))
            
            if self.grayscale:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                #reshape the image to (n,n,1)
                img = img.reshape((img.shape[0], img.shape[1], 1))            

            X_test.append(img)

        X_test = np.array(X_test)
        X_test = np.transpose(X_test, (0, 3, 1, 2))

        return X_test

    def load_data(self, path, test_size = 0.2):
        #read data from csv file
        data = pd.read_csv(path)

        #get the file names and labels
        file_names = data['filename'].values
        labels = data['digit'].values

        print(file_names.shape)
        print(labels.shape)

        print(file_names[0])
        print(labels[0])

        print(type(file_names))
        print(type(labels))

        #make X_train and y_train

        #read the image files and store them in X_train

        X_train = []
        y_train = []

        img_path = path.split(".")[-2]

        for file_name,label in zip(file_names, labels):
            img_name = img_path+"/"+file_name
           
            
            img = cv2.imread(img_name)

            #resize the image to (n,n)
            img = cv2.resize(img, (32,32))
            
            if self.grayscale:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                #reshape the image to (n,n,1)
                img = img.reshape((img.shape[0], img.shape[1], 1))            

            X_train.append(img)
            y = np.zeros(10,)
            y[label] = 1
            y_train.append(y)

        X_train = np.array(X_train)

       
        y_train = np.array(y_train)

        #convert X_train shape from (m, n, n, c) to (m, c, n, n)
        X_train = np.transpose(X_train, (0, 3, 1, 2))

        #normalize the data and convert it to float
        X_train = X_train.astype('float32')
        X_train /= 255

        print(X_train.shape)
        print(y_train.shape)

        X_train, y_train, X_val, y_val = self.split_data(X_train, y_train, test_size = test_size)

        return X_train, y_train, X_val, y_val


    def split_data(self, X, y, test_size = 0.1):
        #split the data into training and validation sets
        m = X.shape[0]
        num_test = math.floor(m*test_size)

        X_train = X[num_test:, :]
        y_train = y[num_test:, :]

        X_val = X[:num_test, :]
        y_val = y[:num_test, :]

        return X_train, y_train, X_val, y_val


    def mini_batches(self, X_train, y_train, batch_size):

        m = X_train.shape[0]
        mini_batches = []

        #shuffle the data
        permutation = list(np.random.permutation(m))
        shuffled_X = X_train[permutation, :]
        shuffled_y = y_train[permutation].reshape((m,10))

        #partition the data
        num_complete_minibatches = math.floor(m/batch_size)

        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[k*batch_size : (k+1)*batch_size, :]
            mini_batch_y = shuffled_y[k*batch_size : (k+1)*batch_size, :]

            mini_batch = (mini_batch_X, mini_batch_y)
            mini_batches.append(mini_batch)

        if m % batch_size != 0:
            mini_batch_X = shuffled_X[num_complete_minibatches*batch_size : m, :]
            mini_batch_y = shuffled_y[num_complete_minibatches*batch_size : m, :]

            mini_batch = (mini_batch_X, mini_batch_y)
            mini_batches.append(mini_batch)\

        #do batch normalization
        # i = 0
        # for mini_batch in mini_batches:
        #     mini_batch_X = mini_batch[0]
        #     mini_batch_y = mini_batch[1]

        #     mean = np.mean(mini_batch_X, axis = 0)
        #     std = np.std(mini_batch_X, axis = 0)

        #     mini_batch_X = (mini_batch_X - mean)/std

        #     mini_batches[i] = (mini_batch_X, mini_batch_y)
        #     i = i + 1




        return mini_batches


        


if __name__ == "__main__":
    data_loader = DataLoader()
    # X_train, y_train = data_loader.load_data("datasets/training-b.csv")
    # mini_batches = data_loader.mini_batches(X_train, y_train, 32)

    # print(len(mini_batches))

    # print(mini_batches[0][0].shape)
    # print(mini_batches[0][1].shape)

    # print(mini_batches[-1][0].shape)
    # print(mini_batches[-1][1].shape)

    X_test = data_loader.load_test_data("datasets/testing-b")
    print(X_test.shape)