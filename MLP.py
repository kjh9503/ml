import random
from funcs import *
from numpy.random import randn, rand
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class MLP_mini(object):
    """ implements a deep neural net.
        Users may specify any number of layers.
        net_arch -- consists of a number of neurons in each layer
    """

    def __init__(self, layers, activate=None, lr=0.1, epochs=1000, batch_size=16, \
                 random_seed=random_seed):

        self.lr = lr
        self.epochs = epochs
        self.layers = layers
        self.n_layers = len(layers)
        self.W = []
        self.batch_size = batch_size
        self.random_seed = random_seed

        np.random.seed(self.random_seed)
        # Random initialization with range of weight values (-1,1)
        # A place holder None is used to indicated "unused place".
        self.W = [[None]]  ## the first W0 is not used.
        for layer in range(self.n_layers - 1):
            w = he_init(self.layers[layer + 1], self.layers[layer])
            self.W.append(w)

        # initialize the activation function list with relu() as default except the last layer
        self.activ = [lambda x: relu(x) for _ in range(self.n_layers - 1)]
        self.activ.append(lambda x: sigmoid(x))
        self.activ_prime = [lambda x: relu_d(x) for _ in range(self.n_layers - 1)]
        self.activ_prime.append(lambda x: sigmoid_d(x))

    def forpass(self, X0):  # A0 : layer input
        # z : dot product 결과, a : activation 결과
        Z = [[None]]  # Z0 is not used.
        X = [X0]  # A0 = X0 is used.
        for i in range(1, len(self.W)):
            z = np.dot(self.W[i], X[i - 1])
            Z.append(z)
            a = self.activ[i](z)
            X.append(a)
        return Z, X

    def backprop(self, Z, X, Y):
        # initialize empty lists to save E and dZ
        # A place holder None is used to indicated "unused place".

        m = Y.shape[1]
        dout = [None for x in range(self.n_layers)]
        dZ = [None for x in range(self.n_layers)]

        # Get error at the output layer or the last layer
        last = self.n_layers - 1
        X[last] = np.clip(X[last], 1e-15, 1 - 1e-15)
        dout[last] = - Y * np.log(X[last]) - (1 - Y) * np.log(1 - X[last])  # (1,405)e
        dZ[last] = (- Y / X[last] + (1 - Y) / (1 - X[last])) * self.activ_prime[last](Z[last])
        loss = np.sum(dout[last]) / m

        # Begin from the back, from the next to last layer
        for i in range(self.n_layers - 2, 0, -1):
            dout[i] = np.dot(self.W[i + 1].T, dZ[i + 1])  # E2 = np.dot(dz3, )
            dZ[i] = dout[i] * self.activ_prime[i](Z[i])

        # Adjust the weights
        m = Y.shape[1]  # number object samples
        for i in range(last, 0, -1):
            grad = np.dot(dZ[i], X[i - 1].T)
            self.W[i] -= self.lr * grad / m

        return loss

    def fit(self, X, y):
        self.loss_ = []
        self.acc_ = []
        self.m_samples = len(y)
        batch_size = self.batch_size
        quo = self.m_samples // batch_size
        rem = self.m_samples % batch_size
        if rem == 0:
            quo -= 1
            rem += batch_size

        for epoch in tqdm(range(self.epochs)):
            mini_batch_loss = []
            for sample in range(quo + 1):  # 예 : 405 // 16 : 25
                if sample == quo:
                    X_train = X[sample * batch_size: sample * batch_size + rem]
                    y_train = y[sample * batch_size: sample * batch_size + rem]
                else:
                    X_train = X[sample * batch_size: (sample + 1) * batch_size]
                    y_train = y[sample * batch_size: (sample + 1) * batch_size]
                X0 = np.array(X_train, ndmin=2).T  # A0 : inputs, minimum 2d array
                Y0 = np.array(y_train, ndmin=2).T  # Y: targets
                Z, x = self.forpass(X0)  # forward pass
                mini_loss = self.backprop(Z, x, Y0)  # back propagation
                mini_batch_loss.append(mini_loss)
            train_acc = self.evaluate(X, y)
            self.acc_.append(train_acc)
            mini_batch_loss = np.mean(mini_batch_loss)
            self.loss_.append(mini_batch_loss)

        epochs_space = np.arange(1, self.epochs + 1)
        plt.plot(epochs_space, self.loss_)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.show()
        print(train_acc)
        plt.plot(epochs_space, self.acc_)
        plt.xlabel('epochs')
        plt.ylabel('train_acc')
        plt.show()
        return self

    def predict_(self, X):  # used in evaluate()
        X0 = np.array(X, ndmin=2).T  # A0: inputs
        Z, X = self.forpass(X0)  # forpass
        result = np.squeeze(X[-1])  # (1,405)
        y_pred = np.array([1 if i >= 0.5 else 0 for i in result])
        return y_pred

    def evaluate(self, Xtest, ytest):  # fully vectorized calculation
        m_samples = len(ytest)
        scores = 0
        y_pred = self.predict_(Xtest)
        for i in range(len(y_pred)):
            if y_pred[i] == ytest[i]:
                scores += 1
        return round(scores / m_samples * 100, 3)