#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Author: HUANG Jingyi
File: network3.py
Date: 2023/11/13 17:51
Email: pierrehuang1998@gmail.com
Description:
"""

import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

class NN:
    def __init__(self, weight_init='random', activation='relu'):
        np.random.seed(0)
        if weight_init == 'random':
            self.w1 = np.random.randn(2, 30)
            self.w2 = np.random.randn(30, 3)
        elif weight_init == 'zeros':
            self.w1 = np.zeros((2, 30))
            self.w2 = np.zeros((30, 3))
        elif weight_init == 'ones':
            self.w1 = np.ones((2, 30))
            self.w2 = np.ones((30, 3))

        # 选择激活函数
        if activation == 'relu':
            self.activation = self.relu
        elif activation == 'sigmoid':
            self.activation = self.sigmoid
        elif activation == 'tanh':
            self.activation = self.tanh

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def call(self, input):
        self.hiden_layer1 = self.activation(input.dot(self.w1))
        self.output = self.hiden_layer1.dot(self.w2)
        self.s = np.exp(self.output)
        self.sm2 = np.sum(self.s, axis=1)
        self.sm2 = np.expand_dims(self.sm2, axis=1)
        self.s = np.divide(self.s, self.sm2)

        # Store the activation values of the hidden layer
        self.hidden_layer_activations = self.hiden_layer1

        return self.s

    def fit(self, data, y, learning_rate=1e-3, epochs=1, method='fixed', batch_size=32):
        loss = np.ones(epochs) * 100
        epoch = 0
        initial_learning_rate = learning_rate
        data_arr = data.shape[0]

        while min(loss) > 10 and epoch < epochs:
            # Split data into mini-batches based on batch_size
            for i in range(0, data_arr, batch_size):
                x_train1 = data[i:i + batch_size]
                y_train1 = y[i:i + batch_size]

                y_pred = self.call(x_train1)
                delta = 1e-7
                loss[epoch] = -np.sum(y_train1 * np.log(y_pred + delta))

                # Backpropagation
                dz2 = self.s - y_train1
                dw2 = self.hiden_layer1.T.dot(dz2)
                da1 = dz2.dot(self.w2.T)
                dz1 = (1 - self.hiden_layer1) * self.hiden_layer1 * da1
                dw1 = x_train1.T.dot(dz1)

                if method == 'variable':
                    # Update learning rate
                    learning_rate = initial_learning_rate * (1 - epoch*10 / epochs)
                    print("Learning rate:", learning_rate)

                self.w1 -= learning_rate * dw1
                self.w2 -= learning_rate * dw2

                epoch += 1

        loss = loss[0:epoch]
        print("Epoch:", epoch - 1)
        return loss




def read_data_x(name):
    x1 = []
    x2 = []
    file = open(name)
    for line in file.readlines():
        curline = line.strip().split(" ")
        x1.append(float(curline[0]))
        x2.append(float(curline[1]))
    x1 = np.array(x1)
    x2 = np.array(x2)
    x1 = np.expand_dims(x1, axis=1)
    x2 = np.expand_dims(x2, axis=1)
    return np.concatenate((x1, x2), axis=1)

def read_data_y(name):
    y = []
    file = open(name)
    for line in file.readlines():
        y.append(int(line[0]))
    y = np.array(y)
    return y

def labelToVec(y):
    y3 = []
    for label in y:
        if label == 1:
            y3.append([1, 0, 0])
        elif label == 2:
            y3.append([0, 1, 0])
        else:
            y3.append([0, 0, 1])
    return y3

def plot_y(x_test, out):
    y1 = out[:,0]
    y2 = out[:,1]
    y3 = out[:,2]
    ax = plt.axes(projection='3d')
    xline = x_test[:,0]
    yline = x_test[:,1]
    ax.scatter3D(xline, yline, y3, c='r', marker='o')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Output")
    plt.show()


def visualize_neurons(nn, x_test):
    # Visualize neurons with activation greater than zero (ai,k > 0)
    active_neurons = np.where(nn.hidden_layer_activations > 0)
    # plt.scatter(x_test[:, 0], x_test[:, 1], c='b', marker='o', label='Data')
    plt.scatter(x_test[active_neurons, 0], x_test[active_neurons, 1], c='r', marker='o', label='Active Neurons (ai,k > 0)')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.title("Visualization of Active Neurons")
    plt.show()

def main():
    x = read_data_x("data_x.txt")
    y = read_data_y("data_y.txt")
    y3 = labelToVec(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y3, test_size=0.2, random_state=37)

    # 可以在这里选择初始值策略
    weight_init = 'random'  # 可以选择 'random'、'zeros'、'ones' 或其他支持的初始值策略
    activation = 'sigmoid'  # 可以选择 'relu'、'sigmoid'、'tanh' 或其他支持的激活函数
    mymodel = NN(weight_init=weight_init, activation=activation)
    epochs = 10000
    learning_rate = 1e-4

    method = 'fixed'  # 'fixed' or 'variable'
    batch_size = 32

    loss = mymodel.fit(x_train, y_train, learning_rate, epochs, method=method, batch_size=batch_size)
    out = mymodel.call(x_test)

    out3 = np.argmax(out, axis=1)
    y_test3 = np.argmax(y_test, axis=1)
    accuracy = accuracy_score(out3, y_test3)
    print("Accuracy:", accuracy)
    visualize_neurons(mymodel, x_test)
    plot_y(x_test, out)

    # Plot the log loss against epochs
    plt.figure()
    plt.plot(np.log(loss), label='Log Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Log Loss')
    plt.title('Log Loss vs. Epochs')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()