import numpy as np

'''
활성화홤수(activation function)

1. step function
2. sigmoid function
3. ReLU function
'''


def step_function(x):
    y = x > 0
    return y.astype(np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLu(x):
    return x