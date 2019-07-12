import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from .layer import Layer
from multiprocessing import Process, Lock, Queue, Pool
import time


class NeuralNetwork:
    """
    Represents a neural network.
    """

    def __init__(self):
        self._layers = []
        self.errors = []
        self.time = 0
        self.stop = False;

    def get_time(self):
        return self.time

    def get_errors(self):
        return self.errors

    def add_layer(self, layer):
        """
        Adds a layer to the neural network.
        :param Layer layer: The layer to add.
        """

        self._layers.append(layer)

    def feed_forward(self, X):
        """
        Feed forward the input through the layers.
        :param X: The input values.
        :return: The result.
        """
        try:

            for layer in self._layers:
                X = layer.activate(X)
        except:
            return;

        return X

    def predict(self, X):
        """
        Predicts a class (or classes).
        :param X: The input values.
        :return: The predictions.
        """
        ff = self.feed_forward(X)

        # One row
        # if ff.ndim == 1:
        #    return np.argmax(ff)

        # Multiple rows
        # return np.argmax(ff, axis=1)
        return ff

    def backpropagation(self, X, y, learning_rate):
        """
        Performs the backward propagation algorithm and updates the layers weights.
        :param X: The input values.
        :param y: The target values.
        :param float learning_rate: The learning rate (between 0 and 1).
        """

        # Feed forward for the output
        output = self.feed_forward(X)

        # Loop over the layers backward
        for i in reversed(range(len(self._layers))):
            layer = self._layers[i]

            # If this is the output layer
            if layer == self._layers[-1]:
                layer.error = y - output
                # The output = layer.last_activation in this case
                layer.delta = layer.error * layer.apply_activation_derivative(output)
            else:
                next_layer = self._layers[i + 1]
                layer.error = np.dot(next_layer.weights, next_layer.delta)
                layer.delta = layer.error * layer.apply_activation_derivative(layer.last_activation)

        # Update the weights
        for i in range(len(self._layers)):
            layer = self._layers[i]
            # The input is either the previous layers output or X itself (for the first hidden layer)
            input_to_use = np.atleast_2d(X if i == 0 else self._layers[i - 1].last_activation)
            layer.weights += layer.delta * input_to_use.T * learning_rate

    def train(self, X, y, learning_rate, max_epochs, pipe_data=None):
        """
        Trains the neural network using backpropagation.
        :param X: The input values.
        :param y: The target values.
        :param float learning_rate: The learning rate (between 0 and 1).
        :param int max_epochs: The maximum number of epochs (cycles).
        :return: The list of calculated MSE errors.
        """
        try:
            self.errors.clear()
            start = time.time()
            for i in range(max_epochs):
                for j in range(len(X)):
                    self.backpropagation(X[j], y[j], learning_rate)
                if i % 10 == 0:
                    res = self.feed_forward(X)
                    mse = np.mean(np.square(y - res))
                    self.errors.append(mse)
                    pipe_data.emit(X,res, mse)
                    if self.stop is True:
                        break
            end = time.time()
            self.time = float(end - start)
        except:
            return

    @staticmethod
    def accuracy(y_pred, y_true):
        """
        Calculates the accuracy between the predicted labels and true labels.
        :param y_pred: The predicted labels.
        :param y_true: The true labels.
        :return: The calculated accuracy.
        """
        if len(y_pred) < len(y_pred):
            return np.abs(np.average(y_pred[:len(y_pred)] - y_true[:len(y_pred)]))
        else:
            return np.abs(np.average(y_pred[:len(y_true)] - y_true[:len(y_true)]))



