import numpy as np
from Layers import Base
import math

class FullyConnected(Base.BaseLayer):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self._gradient_weights = None
        self._optimizer = None
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.weights = np.random.random([self.input_size + 1, self.output_size])
        self.input_tensor = None
        self.input_with_bias = None
        self.error_tensor = None

    def forward(self, input_tensor):
        bias = np.ones(np.size(input_tensor, 0))
        input_with_bias = np.column_stack((input_tensor, bias))
        self.input_tensor = input_tensor
        self.input_with_bias = input_with_bias
        return np.dot(input_with_bias, self.weights)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opti):
        self._optimizer = opti

    def backward(self, error_tensor):
        self.error_tensor = error_tensor
        new_err_tensor = np.dot(error_tensor, self.weights.T)
        self.gradient_weights = np.dot(self.input_with_bias.T, error_tensor)
        if self._optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        return new_err_tensor[:, :-1]

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, grad):
        self._gradient_weights = grad

