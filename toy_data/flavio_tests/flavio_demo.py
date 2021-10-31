#%%
import pandas as pd
import numpy as np
import scipy as sp
import scipy.integrate as integrate
import matplotlib.pyplot as plt

from pynverse import inversefunc

import flavio
# %%
plt.rcParams.update({'text.usetex':False})
# %%
''' Inverse transform method '''
a, b = -1, 3

''' Algorithm

1. generate the cdf of the function
'''

def f(x):
    return np.sin(x) ** 2 / 1.84253


def cdf_single(x):
    return integrate.quad(f, a, x)[0]

def cdf(x):
    return np.array([integrate.quad(f, a, val)[0] for val in x])
# %%
x = np.linspace(a,b)
y = np.linspace(0,1)
plt.plot(x, f(x))
plt.plot(x, cdf(x))
plt.plot(y, inversefunc(f, y_values=y))
# %%
# %%
X = np.random.random(100)
# %%
# %%
y
# %%
inversefunc(f, y_values=[y])
# %%
t = np.linspace(0, 1, 1000)
inv = inversefunc(cdf_single, t)
# %%
plt.plot(t, inv)
# %%
inversefunc(lambda x: x**2, y_values=x)
# %%
