import numba
import numpy as np
import scipy
from numba import njit
from scipy import interpolate
N = 101
x1 = np.linspace(0,10,N)
y1 = x1**2 + np.random.randn(N)*0.001
z1 = 5

@njit
def f(x,y, z) :
    ff = interpolate.interp1d(x,y,kind='cubic')
    return ff(z)
f(x1,y1,z1)