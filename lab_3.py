#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import math
from collections import Counter

x = sp.symbols('x')
expression = sp.exp(x)
y_function = sp.lambdify(x, expression, 'numpy')
a = -7
b = 5
a_x = np.exp(a)
b_x = np.exp(b)
N_HIKVADRAT = 200
N_KOLM = 30
N_MIZ = 50
HIKVADRAT = 15.086
KOLM = 1.63
MIZ = 0.744
h = []
A = []
B = []
v = []
f = []
p = []
p_v = []
xi_kvad = []


def elements_in_interval(start, end):
    count = 0
    for i in temp:
        if i > end:
            break
        if start <= i:
            count += 1
    return count


def create_sample(n, a, b):
    x = []
    y = []
    x_random = np.random.uniform(0, 1, n)
    for i in range(n):
        x.append(round(x_random[i] * (b - a) + a, 3))
        y.append(round(y_function(x[i]), 3))
    return x, y


def equal_probability():
    for i in range(0, M):
        v.append(int(N_HIKVADRAT / M))
    A.append(round(temp[0], 5))
    for i in range(1, M):
        A.append(round((temp[i * v[i] - 1] + temp[i * v[i]]) / 2, 5))
    B.extend(A[1:])
    B.append(round(temp[-1], 5))

    for i in range(0, M):
        h.append(round(B[i] - A[i], 5))
        f.append(round(v[i] / (N_HIKVADRAT * h[i]), 10))


def create_table(check):
    table = PrettyTable()
    if check == 1:
        table.field_names = ['A', 'B', 'h', 'v', 'f']
        for i in range(0, M):
            table.add_row([A[i], B[i], h[i], v[i], f[i]])
    elif check == 2:
        table.field_names = ['F(A)', 'F(B)', 'p', 'p*', 'X^2']
        for i in range(0, M):
            table.add_row([y_function(A[i]), y_function(B[i]), p[i], p_v[i], xi_kvad[i]])
    return table


def hist(mes):
        plt.grid(True)
        plt.title(mes)
        for i in range(0, M):
            plt.bar(A[i], f[i], width=h[i], align='edge')


def polygon():
        for i in range(0, M - 1):
            plt.plot([(B[i] + A[i]) / 2, (B[i + 1] + A[i + 1]) / 2], [f[i], f[i + 1]], color='r', marker='.')


def theoretical_func(x_local):
        return (np.log(x_local) - a) * 1/(b-a)

def theoretical_func_graphic(a, b):
    x = np.arange(a, b, 0.1)
    y = np.arange(a, b, 0.1)
    for i in range(len(x)):
        y[i] = (np.log(x[i]) + 5) / 12
    plt.plot(x, y, color='r')
    plt.grid(True)



def theor_f():
    x = np.arange(np.exp(a), np.exp(b), 0.1)
    y = 1 / (12 * x)
    plt.plot(x, y, color='yellow')


def theoretical_probability(M):
    for i in range(M):
        p.append(theoretical_func(B[i]) - theoretical_func(A[i]))
        p_v.append(v[i] / N_HIKVADRAT)
        xi_kvad.append((N_HIKVADRAT * (p[i] - p_v[i]) ** 2) / p[i])
    return sum(xi_kvad), sum(p)


def xi_kvadrat():

    sum_xi_kvad, lol_p = theoretical_probability(M)
    print(create_table(2))
    print(f'Хи квадрат = {sum_xi_kvad} \t {HIKVADRAT}')


def empirical_func(x, y):
    cur = -10e9
    ind = 0
    for i in range(len(y)):
        if cur >= x:
            break
        cur = y[i]
        ind += 1
    return (ind - 1) / len(y)

def empirical_func_graphic(sample):

    n = sum(sample.values())
    k = sorted(sample.keys())
    p = 0
    plt.plot([0, k[0]], [0, 0], marker='.')
    for i in range(1, len(c)):
        p += c.get(k[i - 1]) / n

        plt.plot([k[i - 1], k[i]], [p, p], marker='.')
    plt.grid(True)




def kolmogorov():
    x, y = create_sample(N_KOLM, a, b)
    y.sort()

    cur = 0.0
    for i in range(0, len(y)):
        cur = max(abs(empirical_func(y[i], y) - theoretical_func(y[i])), cur)
    #print(f'max|F*(x) - F0(x)| = {cur}')
    print(f'Критерий Колмогорова" = {np.sqrt(len(y)) * cur}\t{KOLM}')


def mises():
    x, y = create_sample(N_MIZ, a, b)
    y.sort()
    temp = []

    for i in range(1, len(y)):
        temp.append((theoretical_func(y[i]) - (i - 0.5) / len(y)) ** 2)
    C = 1 / (12 * len(y))
    for tmp in temp:
        C += tmp
    print(f'Критерий Мизеса = {C}\t{MIZ}')





X, temp = create_sample(N_HIKVADRAT, a, b)
temp = sorted(temp)
c = Counter(temp)

if N_HIKVADRAT <= 100:
    M = int(np.sqrt(N_HIKVADRAT))
else:
    M = int(4 * math.log10(N_HIKVADRAT))

equal_probability()

print(create_table(1))

hist("Гистограмма равновероятностная методам")
#polygon()
theor_f()
ylimit = 0.08
plt.ylim(top=ylimit)
plt.show()

empirical_func_graphic(c)
theoretical_func_graphic(a_x, b_x)
plt.show()

xi_kvadrat()
kolmogorov()
mises()
