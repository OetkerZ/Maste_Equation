import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from numpy.linalg import inv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
Nl = 10
Ca = 1 * pow(10, -16)
e = 1.602 * pow(10, -19)
kb = 1.38064852 * pow(10, -23)
Ec = e * e / (2 * Ca)
T = 3
gamma = 100000000
pixel = 250
# Nl = 15
# Ca = 0.01
# e = 1
# kb = 1
# Ec = e * e / (2 * Ca)
# T = 4
# gamma =0.001
# pixel = 500


def muL(V):
    return (e * V / 2)


def muR(V):
    return (-e * V / 2)


def EN(n, Vg):
    return Ec * n * n - e * Vg * n


def mu(n1, n2, Vg):
    return EN(n1, Vg) - EN(n2, Vg)


def Nopt(Vg):
    return floor(e * Vg / 2 / Ec)


def gammaf(E, m):
    return 1 / (np.exp((E - m) / (kb * T)) + 1)


def gammafe(E):
    return (E) / (np.exp((E) / (kb * T)) - 1)


def W(n1, n2, V, Vg):
    if n1 < n2:
        return np.sum([1 - gammaf(mu(n2, n1, Vg), muL(V)), 1 - gammaf(mu(n2, n1, Vg), muR(V))])
    else:
        return np.sum([gammaf(mu(n1, n2, Vg), muL(V)), gammaf(mu(n1, n2, Vg), muR(V))])
    # return gammafe(mu(n1,n2,Vg)+(n1-n2)*muL(V))+gammafe(mu(n1,n2,Vg)+(n1-n2)*muR(V))

# def Wn(n, V, Vg):
#     data = np.array([0])
#     row = np.array([0])
#     col = np.array([0])
#     for i in range(n - 1):
#         row = np.append(row, [i + 1])
#         col = np.append(col, [i + 1])
#         data = np.append(data, [-W(i + 2, i + 1, V, Vg) - W(i, i + 1, V, Vg)])
#     for j in range(n - 1):
#         row = np.append(row, [j])
#         col = np.append(col, [j + 1])
#         data = np.append(data, [W(i, i + 1, V, Vg)])
#     for k in range(n - 1):
#         row = np.append(row, [k + 1])
#         col = np.append(col, [k])
#         data = np.append(data, [W(i + 1, i, V, Vg)])

#     return coo_matrix((data, (row, col)), shape=(n, n)).toarray()


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
        data = np.append(data, [W(k+ 1, k, V, Vg)])

    return coo_matrix((data, (row, col)), shape=(n, n)).toarray()


# def Pn(n, WMAT):
#     dim = np.arange(n * n)
#     dim = dim.reshape(n, n)
#     One = np.ones_like(dim)
#     R = np.ones_like(np.arange(n))
#     return np.dot(inv(np.add(One, WMAT)), R)
def Pn(n, WMAT):
    R = np.eye(1, n).flatten()
    return np.linalg.solve(WMAT, R)


def Il(n, Pn, V, Vg):
    I = 0
    i = 0
    for p in Pn:
        I += p * (W(i+1, i, V, Vg) - W(i , i+1, V, Vg))
        i += 1
    return I * (-e * gamma)


def dataP(Pn, dataW, Vg, N):
    data = np.array()
    for W in dataW[Vg]:
        data = np.append(data, np.Pn(N, W))
    return data
def Cond(WMAT,d):
    data=np.array([])
    WMATT=np.transpose(WMAT)
    for i in WMATT:
        data=np.append(data,np.ediff1d(i)/(d/pixel))
    data=data.reshape(pixel,pixel-1)
    data=np.delete(data,pixel-1,0)
    data=np.transpose(data)

    return data.reshape(pixel-1,pixel-1)




Vglist = np.arange(0.006, 0.014, 0.008 / pixel)
Vlist = np.arange(-0.004, 0.004, 0.008 / pixel)
dataWn = np.array([])
dataIl = np.array([])
dataPn = np.array([])
dataC=np.array([])
for i in Vlist:
    dataPn = np.append(dataPn, Pn(Nl, Wn(Nl, i, 0.01)).take(5))
    for j in Vglist:
        # dataWn = np.append((dataWn, [Wn(Nl, i, j))])
        dataIl = np.append(dataIl, Il(Nl, Pn(Nl, Wn(Nl, i, j)), i, j))
dataIl = dataIl.reshape(pixel, pixel)
dataC=Cond(dataIl,0.008)

#fig = plt.figure()
#ax = fig.gca(projection='3d')
#dataWn = dataWn.reshape(pixel, pixel)

X, Y = np.meshgrid(Vglist, Vlist)
#print(dataIl.shape, Vglist, Vlist, dataIl)
#ax.plot_surface(X, Y,dataIl ,cmap=cm.coolwarm, linewidth=0, antialiased=False)

#print(Wn(4,0.00002,0.0003))
#plt.plot(Vglist, dataPn)

#plt.contourf(Vglist,Vlist,dataIl,100)
plt.contourf(np.delete(Vglist,pixel-1),np.delete(Vglist,pixel-1),dataC,100)

plt.show()
