import numpy as np
import matplotlib.pylab as plt

'''
활성화홤수(activation function)
뉴런과 뉴런 사이에 정보 전달에서 100% 그대로 전달하지 않고 정보의 재해석이 발생함
하지만 인공신경망의 경우 선형적으로 그대로 정보를 전달함.
이전 레이어에 있는 노드값이 다음 레이어로 전달되는 경우 비선형으로, 정보의 재해석 기능을 담당
1. step function
입력값을 0 혹은 1로 변환
2. sigmoid function
입력값을 0~1 사이의 확률적으로 표현할 수 있는 값으로 변환
3. ReLU function
4. softmax
입력값을 0~1사이의 K개의 클래스로 분류할 수 있는 값으로 변환
'''


def step_function(x):
    y = x > 0
    return y.astype(np.int)


def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y


def ReLu(x):
    return x

def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y