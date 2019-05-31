#!/usr/bin/python3
# -*- coding: utf-8 -*-


import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from collections import Counter
from prettytable import PrettyTable

def create_sample(n, a, b):
    X = []
    Y = []
    Xi = np.random.uniform(0, 1, n)
    for i in range(n):
        X.append(round(Xi[i] * (b - a) + a, 3))
        Y.append(round(y_function(X[i]), 3))
    return X, Y


def empirical_func(sample):
    #print(sample)

    n = sum(sample.values())
    k = sorted(sample.keys())
    p = 0


    plt.plot([0, k[0]], [0, 0], marker='.')
    for i in range(1, len(c)):
        p += c.get(k[i - 1]) / n

        plt.plot([k[i - 1], k[i]], [p, p], marker='.')
    # plt.plot([k[len(c)-1], k[len(c)-1] + 2], [1, 1], marker='o')
    plt.grid(True)
    #theoretical_func(0.1, 36.0)

    #plt.show()


def theoretical_func(a, b):
    x = np.arange(a, b, 0.1)
    #y = (x * ((np.log(x) + 5) / 12) / x)
    y = np.arange(a, b, 0.1)
    for i in range(len(x)):
        y[i] = (np.log(x[i]) + 5) / 12

    plt.plot(x, y, color='r')

    plt.grid(True)
    #plt.show()


n = int(input('Input n: '))

x = sp.symbols('x')
expression = sp.exp(x)

y_function = sp.lambdify(x, expression, 'numpy')

a = -5
b = 7

a_x = np.exp(-5)
b_x = np.exp(7)

X, Y = create_sample(n, a, b)

table = PrettyTable()
table.field_names = ['x', 'y']
for i in range(len(X)):
    table.add_row([X[i], Y[i]])
print(table)



c = Counter(Y)
print('Вариационный ряд:')
print(sorted(Y))
empirical_func(c)
plt.show()
theoretical_func(a_x, b_x)
plt.show()


empirical_func(c)
theoretical_func(a_x, b_x)
plt.show()



