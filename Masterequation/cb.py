import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from numpy.linalg import inv
Nl = 50
Ca = 10 * pow(10, -15)
e = 1.602 * pow(10, -19)
kb = 1.38064852 * pow(10, -23)
Ec = e * e / (2 * Ca)
T = 3
gamma =1000000000
pixel = 150
# Nl = 15
# Ca = 0.01
# e = 1
# kb = 1
# Ec = e * e / (2 * Ca)
# T = 4
# gamma =0.001
# pixel = 500

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

def gammafe(E):
    return (E) / (np.exp((E) / (kb * T)) - 1)


def W(n1, n2, V, Vg):
    if n1 < n2:
        return np.sum([1 - gammaf(mu(n2, n1, Vg), muL(V)), 1 - gammaf(mu(n2, n1, Vg), muR(V))])
    else:
        return np.sum([gammaf(mu(n1, n2, Vg), muL(V)), gammaf(mu(n1, n2, Vg), muR(V))])
    # return gammafe(mu(n1,n2,Vg)+(n1-n2)*muL(V))+gammafe(mu(n1,n2,Vg)+(n1-n2)*muR(V))

def Wn(n, V, Vg):
    data = np.array([0])
    row = np.array([0])
    col = np.array([0])
    for i in range(n - 1):
        row = np.append(row, [i + 1])
        col = np.append(col, [i + 1])
        data = np.append(data, [-W(i + 2, i + 1, V, Vg) - W(i, i + 1, V, Vg)])
    for j in range(n - 1):
        row = np.append(row, [j])
        col = np.append(col, [j + 1])
        data = np.append(data, [W(i, i + 1, V, Vg)])
    for k in range(n - 1):
        row = np.append(row, [k + 1])
        col = np.append(col, [k])
        data = np.append(data, [W(i + 1, i, V, Vg)])

    return coo_matrix((data, (row, col)), shape=(n, n)).toarray()


def Pn(n, WMAT):
    dim = np.arange(n * n)
    dim = dim.reshape(n, n)
    One = np.ones_like(dim)
    R = np.ones_like(np.arange(n))
    return np.dot(inv(np.add(One, WMAT)), R)


def Il(n, Pn, V, Vg):
    I = 0
    i = 1
    for p in Pn:
        I += p * (W(i, i - 1, V, Vg) - W(i - 1, i, V, Vg))
        i += 1
    return I * (-e * gamma)


def dataP(Pn, dataW, Vg, N):
    data = np.array()
    for W in dataW[Vg]:
        data = np.append(data, np.Pn(N, W))
    return data


Vglist = np.arange(-0.008,0.008, 0.016 / pixel)
Vlist = np.arange(-0.02, 0.02, 0.04/ pixel)
dataWn = np.array([])
dataIl = np.array([])
dataPn = np.array([])

for i in Vlist:
    for j in Vglist:
        #dataWn = np.append((dataWn, [Wn(Nl, i, j))])
        dataIl = np.append(dataIl, Il(Nl, Pn(Nl, Wn(Nl, i, j)), i, j))
        dataPn=np.append(dataPn,Pn(Nl,Wn(Nl, i, j)).take(2))

#dataWn = dataWn.reshape(pixel, pixel)
dataIl = dataIl.reshape(pixel, pixel)
dataPn = dataPn.reshape(pixel,pixel)

plt.contourf(Vglist, Vlist, dataIl)
plt.show()