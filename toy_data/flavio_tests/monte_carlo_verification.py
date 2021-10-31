#%%
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'text.usetex':False})

import flavio
import flavio.plots
import pandas as pd

import matplotlib.ticker as tck


#Supress the warnings about QCDF corrections above 6 being unreliable
import warnings
from numpy.lib.function_base import angle

import wilson
warnings.filterwarnings("ignore")
from tqdm import tqdm
# %%
def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$\frac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'$\frac{-%s}{%s}$'%(latex,den)
            else:
                return r'$\frac{%s%s}{%s}$'%(num,latex,den)
    return _multiple_formatter

class Multiple:
    def __init__(self, denominator=2, number=np.pi, latex='\pi'):
        self.denominator = denominator
        self.number = number
        self.latex = latex

    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)

    def formatter(self):
        return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex))
# %%
def compute_J_from_vec(x):

    return x * np.exp(-x)

# %%
J_min, J_max = 0,1.7
x_min, x_max = 0,7


data_points = []
excluded = 0
included = 0
for i in tqdm(range(500000)):
    # 1. generate random J
    J_rnd = np.random.random() * 2

    # 2. gnerate random kinematic vector
    data_vector = {
        'x':np.random.uniform(x_min, x_max), 
        } # verified to be uniform

    # 3. compute J from `data_vector`
    J_comp = compute_J_from_vec(data_vector['x'])

    # 4. compare to random J
    if J_rnd < J_comp:
        data_vector['$J_comp$'] = J_comp
        data_points.append(data_vector)
        included +=1 
    else:
        excluded +=1 

# %%
data_points = pd.DataFrame(data_points)
# %%
f,ax=plt.subplots(figsize=(5,5))

ax.hist(data_points['x'], bins=100, density=False, label='Generated Data')
x = np.linspace(x_min, x_max, 100)
ax.plot(x, 810*compute_J_from_vec(x)*3, label='$x e^{-x}$')
ax.legend()
ax.set_xlabel('$x$')
plt.show()
# %%
