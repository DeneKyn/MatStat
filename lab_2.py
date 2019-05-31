#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import math

x = sp.symbols('x')
expression = sp.exp(x)
# expression = x**2
y_function = sp.lambdify(x, expression, 'numpy')
a = -7
b = 5
n = 100000
h = []
A = []
B = []
v = []
f = []

def elements_in_interval(start, end):
    cnt = 0
    for i in temp:
        if i > end:
            break
        if start <= i:
            cnt += 1
    return cnt


def create_sample(n, a, b):
    X = []
    Y = []
    Xi = np.random.uniform(0, 1, n)
    for i in range(n):
        X.append(round(Xi[i] * (b - a) + a, 3))
        Y.append(round(y_function(X[i]), 3))
    return X, Y


def equal_interval():
    for i in range(0, M):
        h.append(round((temp[-1] - temp[0]) / M, 5))
        A.append(round(temp[0] + i * h[i], 5))
        B.append(round(temp[0] + (i + 1) * h[i], 5))
        v.append(elements_in_interval(A[i], B[i]))
        f.append(round(v[i] / (n * h[i]), 10))


def equal_probability():
    for i in range(0, M):
        v.append(int(n / M))
    A.append(round(temp[0], 5))
    for i in range(1, M):
        A.append(round((temp[i * v[i] - 1] + temp[i * v[i]]) / 2, 5))
    B.extend(A[1:])
    B.append(round(temp[-1], 5))

    for i in range(0, M):
        h.append(round(B[i] - A[i], 5))
        f.append(round(v[i] / (n * h[i]), 10))


def create_table():
    table = PrettyTable()
    table.field_names = ['A', 'B', 'h', 'v', 'f']
    for i in range(0, M):
        table.add_row([A[i], B[i], h[i], v[i], f[i]])
    return table


def hist(mes):
    plt.grid(True)
    plt.title(mes)
    for i in range(0, M):
        plt.bar(A[i], f[i], width=h[i], align='edge')


def polygon():
    for i in range(0, M - 1):
        plt.plot([(B[i] + A[i]) / 2, (B[i + 1] + A[i + 1]) / 2], [f[i], f[i + 1]], color='r', marker='.')


def theor_f():
    x = np.arange(np.exp(a), np.exp(b), 0.1)
    y = 1 / (12 * x)
    plt.plot(x, y, color='yellow')


def calc_V():
    ans = 0.0
    for i in range(0, M):
        ans += f[i] * h[i]
    return ans

def empirical_func(Y, N, n):
    # group =  [[key, len(list(group))] for key, group in groupby(Y)]
    N = list(map(int, N))
    group = [[yi, ni] for yi, ni in zip(Y, N)]
    group.sort(key=lambda x: x[0])
    group = [[group[0][0] - 0.5, 0]] + group

    XX = [group[0][0]]
    YY = [0]
    for i in range(1, len(group)):
        XX.append(group[i][0])
        XX.append(group[i][0])
        YY.append(YY[2 * (i - 1)])
        YY.append(YY[2 * (i - 1)] + group[i][1] / n)
    XX.append(group[-1][0] + 0.5)
    YY.append(YY[-1])
    return XX, YY


def theoretical_func(a, b):
    x = np.arange(a, b, 0.1)
    #y = (x * ((np.log(x) + 5) / 12) / x)
    y = np.arange(a, b, 0.1)
    for i in range(len(x)):
        y[i] = (np.log(x[i]) - a) * 1/(b-a)

    plt.plot(x, y, color='r')

X, temp = create_sample(n, a, b)
temp = sorted(temp)

if n <= 100:
    M = int(np.sqrt(n))
else:
    M = int(3 * math.log10(n))
equal_interval()

print(create_table())


hist("Гистограмма равноинтервальным методам")
polygon()
theor_f()
ylimit = max(f) + max(f) * 0.10
plt.ylim(top=ylimit)
print(calc_V())
plt.show()

#pol_x1 = [(a + b) / 2 for a, b in zip(A, B)]
#XX, YY = empirical_func(pol_x1, v, n)
#plt.plot(XX, YY)
#theoretical_func(np.exp(a), np.exp(b))
#plt.show()


h.clear()
A.clear()
B.clear()
v.clear()
f.clear()

equal_probability()

hist("Гистограмма равновероятностным методам")
polygon()
theor_f()
plt.ylim(top=ylimit)
plt.show()

print(create_table())
print(calc_V())

#pol_x2 = [(a + b) / 2 for a, b in zip(A, B)]
#XX, YY = empirical_func(pol_x2, v, n)
#plt.plot(XX, YY)
#theoretical_func(np.exp(a), np.exp(b))
#plt.show()


