import numpy as np

class Sgd:
    def __init__(self, learning_rate):
        self.learning_rate = float(learning_rate)

    def calculate_update(self, weight_tensor, gradient_tensor):
        result = weight_tensor - self.learning_rate * gradient_tensor
        return np.array(result)


class Sgd:
    def __init__(self, learning_rate):
        self.learning_rate = float(learning_rate)

    def calculate_update(self, weight_tensor, gradient_tensor):
        result = weight_tensor - self.learning_rate * gradient_tensor
        return np.array(result)


## https://blog.csdn.net/csdn_zhishui/article/details/82791114
## https://blog.csdn.net/yzy_1996/article/details/84618536

class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        v = np.zeros_like(weight_tensor)
        next_w = None
        v = self.momentum_rate * v - self.learning_rate * gradient_tensor
        next_w = weight_tensor + v

        return next_w


class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho

    def calculate_update(self, weight_tensor, gradient_tensor):
        m = np.zeros_like(weight_tensor)
        v = np.zeros_like(weight_tensor)
        t = 1
        epsilon = 1e-8
        beta1 = self.mu
        beta2 = self.rho

        m = beta1 * m + (1 - beta1) * gradient_tensor
        v = beta2 + v + (1 - beta2) * (gradient_tensor ** 2)
        mb = m / (1 - beta1 ** t)
        vb = v / (1 - beta2 ** t)
        next_w = weight_tensor - self.learning_rate * mb / (np.sqrt(vb) + epsilon)

        return next_w
