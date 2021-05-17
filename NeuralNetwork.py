import numpy as np
from copy import deepcopy


class NeuralNetwork:
    def __init__(self, optimizer):
        self._optimizer = optimizer
        self.loss = list()
        self.layers = list()
        self.data_layer = None
        self.loss_layer = None
        self.label_tensor = None

    def forward(self):
        activation_tensor, self.label_tensor = self.data_layer.next()
        for layer in self.layers:
            activation_tensor = layer.forward(activation_tensor)
        return self.loss_layer.forward(activation_tensor, self.label_tensor)

    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)
        return error_tensor

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = deepcopy(self._optimizer)
        self.layers.append(layer)

    def train(self, iterations):
        for _ in range(iterations):
            tmp = self.forward()
            self.backward()
            self.loss.append(tmp)

    def test(self, input_tensor):
        input_t = np.copy(input_tensor)
        for layer in self.layers:
            input_t = layer.forward(input_t)
        return input_t

