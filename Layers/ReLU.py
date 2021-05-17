import numpy as np
from Layers import Base


class ReLU(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_tensor = None
        self.error_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return 0.5 * (np.abs(input_tensor) + input_tensor)

    def backward(self, error_tensor):
        self.error_tensor = error_tensor
        tmp = self.input_tensor / np.abs(self.input_tensor)
        mask = 0.5 * (np.abs(tmp) + tmp) * tmp
        return np.multiply(mask, error_tensor)

