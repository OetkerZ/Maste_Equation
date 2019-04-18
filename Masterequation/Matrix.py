import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from numpy.linalg import inv
Nl = 4
Ca = 10 * pow(10, -16)
e = 1.602 * pow(10, -19)
kb = 1.38064852 * pow(10, -23)
Ec = e * e / (2 * Ca)
T = 300
gamma = 1000000000
pixel = 160


def muL(V):
    return e * V / 2


def muR(V):
    return -e * V / 2


def EN(n, Vg):
    return Ec * n * n - e * Vg * n


def mu(n1, n2, Vg):
    return EN(n1, Vg) - EN(n2, Vg)


def Nopt(Vg):
    return floor(e * Vg / 2 / Ec)


def gammaf(E, m):
    return 1 / (np.exp((E - m) / kb / T) + 1)
    #E / (np.exp(E / (kb * T)) - 1)#


def W(n1, n2, V, Vg):
    if n1 < n2:
        return np.sum([1 - gammaf(mu(n2, n1, Vg), muL(V)), 1 - gammaf(mu(n2, n1, Vg), muR(V))])
    else:
        return np.sum([gammaf(mu(n2, n1, Vg), muL(V)), gammaf(mu(n2, n1, Vg), muR(V))])


def Wn(n, V, Vg):
    data = np.ones_like(np.arange(n))
    row = np.zeros(n)
    col = np.arange(0, n)
    for i in range(n - 1):
        row = np.append(row, [i + 1])
        col = np.append(col, [i + 1])
        data = np.append(data, [-W(i + 2, i + 1, V, Vg) - W(i, i + 1, V, Vg)])
    for j in range(n - 2):
        row = np.append(row, [j + 1])
        col = np.append(col, [j + 2])
        data = np.append(data, [W(j, j + 1, V, Vg)])
    for k in range(n - 1):
        row = np.append(row, [k + 1])
        col = np.append(col, [k])
        data = np.append(data, [W(k + 1, k, V, Vg)])
    print(data,row,col)
    return coo_matrix((data, (row, col)), shape=(n, n)).toarray()


print(Wn(Nl, 0.00002, 0.0003),W(1,0,0.00002,0.0003))
