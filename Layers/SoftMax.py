import numpy as np
from Layers import Base


class SoftMax(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_tensor = None
        self.output_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        max_arr = np.tile(input_tensor.max(1), (input_tensor.shape[1], 1)).T
        tmp = np.exp(input_tensor - max_arr)
        sum_exp = np.tile(np.sum(tmp, axis=1), (input_tensor.shape[1], 1)).T
        self.output_tensor = np.divide(tmp, sum_exp)
        return self.output_tensor

    def backward(self, error_tensor):
        tmp = np.multiply(error_tensor, self.output_tensor)
        tmp2 = np.tile(np.sum(tmp, axis=1), (error_tensor.shape[1], 1)).T
        return np.multiply(self.output_tensor, (error_tensor - tmp2))


