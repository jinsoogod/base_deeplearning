import numpy as np
from actviation_function import sigmoid
from actviation_function import softmax
'''
input = x1, x2, bias(3개)
hidden layer1 = a1, a2, a3 (3개)
hidden layer2 = a4, a5, a6 (2개)
output layer = y1, y2 (2개)
'''

# hidden layer1
'''
a1 = XW + B
X = x1,x2
W = w11, w12
B = b1

a2 = XW + B
X = x1, x2
W = w21, w22
B = b2

a3 = XW + B
X = x1, x2
W = w31, w32
B = b3

일반화
A = XW + B
A = a1, a2 ,a3
W = w11, w12, w13
       w21, w22, w23
X = x1, x2
B = b1, b2 ,b3
'''
x1 = 1.0
x2 = 0.5
w11 = 0.1
w12 = 0.3
w13 = 0.5
w21 = 0.2
w22 = 0.4
w23 = 0.6
b1 = 0.1
b2 = 0.2
b3 = 0.3

a1 = (x1 * w11) + (x2 * w21) + b1
A1 = sigmoid(a1)
a2 = (x1 * w12) + (x2 * w22) + b2
A2 = sigmoid(a2)
a3 = (x1 * w13) + (x2 * w23) + b3
A3 = sigmoid(a3)


# hidden layer2
'''
a4 = XW + B
X = A1, A2, A3
W = A1w4, A2w4, A3w4
B = b4

a5 = XW + B
X = A1, A2 ,A3
W = A1w5, A2w5, A3w5
B = b5

a6 = XW + B
X = A1, A2, A3
W = A1w6, A2w6, A3w6
B = b6

일반화
A = XW + B
A = A4, A5, A6
W = A1w4, A2w4, A3w4,
        A1w4, A2w4, A3w4,
        A1w4, A2w4, A3w4
X = A1, A2 ,A3
B = b4, b5, b6
'''
A1w4 = 0.1
A2w4 = 0.4
A3w4 = 0.7
b4 = 0.1
a4 = (A1*A1w4) + (A2*A2w4) + (A3*A3w4) + b4
A4 = sigmoid(a4)

A1w5 = 0.2
A2w5 = 0.5
A3w5 = 0.8
b5 = 0.2
a5 = (A1*A1w5) + (A2*A2w5) + (A3*A3w5) + b5
A5 = sigmoid(a5)

A1w6 = 0.3
A2w6 = 0.6
A3w6 = 0.9
b6 = 0.3
a6 = (A1*A1w6) + (A2*A2w6) + (A3*A3w6) + b6
A6 = sigmoid(a6)

# output layer
'''
y1 = XW + B
X = A4, A5, A6
W = A4w1, A5w1, A6w1
B = b7

y2 = XW + B
X = A4, A5 ,A6
W = A4w2, A5w2, A6w2
B = b8

일반화
A = XW + B
A = y1, y2
W = A4w1, A5w1, A6w1,
        A4w2, A5w2, A6w2
X = A4, A5 ,A6
B = b7, b8
'''
A4w1 = 0.1
A5w1 = 0.3
A6w1  = 0.5
b7 = 0.1
y1 = (A4*A4w1) + (A5*A5w1) + (A6*A6w1) + b7

A4w2 = 0.2
A5w2 = 0.4
A6w2  = 0.6
b8 = 0.2
y2 = (A4*A4w2) + (A5*A5w2) + (A6*A6w2) + b8

softmax_input = np.array([y1, y2])
output = softmax(softmax_input)

print("End")