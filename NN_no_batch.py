from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


class NN:
    def __init__(self, activation='relu'):
        np.random.seed(0)
        self.activation = activation

        # Initialize weights based on activation function
        if self.activation == 'relu':
            self.w1 = np.random.randn(2, 30)
            self.w2 = np.random.randn(30, 3)
        elif self.activation == 'sigmoid':
            self.w1 = np.zeros((2, 30))
            self.w2 = np.zeros((30, 3))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def activate(self, x):
        if self.activation == 'sigmoid':
            return self.sigmoid(x)
        elif self.activation == 'relu':
            return self.relu(x)

    def call(self, input):
        self.hiden_layer1 = self.activate(input.dot(self.w1))
        self.output = self.hiden_layer1.dot(self.w2)
        self.s = np.exp(self.output)
        self.sm2 = np.sum(self.s, axis=1)
        self.sm2 = np.expand_dims(self.sm2, axis=1)
        self.s = np.divide(self.s, self.sm2)
        return self.s

    def fit(self, data, y, learning_rate=1, epochs=1):
        loss = np.zeros(epochs)
        v1 = 0
        v2 = 0
        ro = 0.9

        for epoch in range(epochs):
            y_pred = self.call(data)
            delta = 1e-7
            loss[epoch] = -np.sum(y * np.log(y_pred + delta))
            dz2 = self.s - y
            dw2 = self.hiden_layer1.T.dot(dz2)
            da1 = dz2.dot(self.w2.T)
            dz1 = (1 - self.hiden_layer1) * self.hiden_layer1 * da1
            dw1 = data.T.dot(dz1)

            # Apply gradient descent updates with momentum
            # v1 = ro * v1 + dw1
            self.w1 -= learning_rate * dw1
            # v2 = ro * v2 + dw2
            self.w2 -= learning_rate * dw2

        return loss


def read_data_x(name):
    x1 = []
    x2 = []
    file = open(name)
    for line in file.readlines():
        curline = line.strip().split(" ")
        x1.append(curline[0])
        x2.append(curline[1])
    x1 = np.array(x1, dtype=float)
    x2 = np.array(x2, dtype=float)
    x1 = np.expand_dims(x1, axis=1)
    x2 = np.expand_dims(x2, axis=1)
    print("X1:", x1.shape)
    print("x2", x2.shape)
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
        else:
            if label == 2:
                y3.append([0, 1, 0])
            else:
                y3.append([0, 0, 1])
    return y3


# 读取数据
x = read_data_x("data_x.txt")
y = read_data_y("data_y.txt")
y3 = labelToVec(y)

# 整个数据集划分为训练集与测试集
x_train, x_test, y_train, y_test = train_test_split(x, y3, test_size=0.8, random_state=37)
print("x:", x_train.shape)

# 数据标准化
ss_x = StandardScaler()
x_train = ss_x.fit_transform(x_train)
x_test = ss_x.transform(x_test)

# 训练的超参数
epochs = 150
learning_rate = 1e-4

# 使用 ReLU 激活函数的模型
mymodel_relu = NN(activation='relu')
loss_relu = mymodel_relu.fit(x_train, y_train, learning_rate, epochs)
out_relu = mymodel_relu.call(x_test)
out3_relu = np.argmax(out_relu, axis=1)
y_test3 = np.argmax(y_test, axis=1)
accuracy_relu = accuracy_score(out3_relu, y_test3)
print("ReLU Activation Accuracy:", accuracy_relu)

# 使用 Sigmoid 激活函数的模型
mymodel_sigmoid = NN(activation='sigmoid')
loss_sigmoid = mymodel_sigmoid.fit(x_train, y_train, learning_rate, epochs)
out_sigmoid = mymodel_sigmoid.call(x_test)
out3_sigmoid = np.argmax(out_sigmoid, axis=1)
accuracy_sigmoid = accuracy_score(out3_sigmoid, y_test3)
print("Sigmoid Activation Accuracy:", accuracy_sigmoid)

import matplotlib.pyplot as plt

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_relu)
plt.xlabel('Epoch number')
plt.ylabel('Loss (Cross Entropy)')
plt.title('ReLU Activation')

plt.subplot(1, 2, 2)
plt.plot(loss_sigmoid)
plt.xlabel('Epoch number')
plt.ylabel('Loss (Cross Entropy)')
plt.title('Sigmoid Activation')

plt.tight_layout()
plt.show()
