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
def compute_J_from_vec(vec, wilson_coef=None):
    if wilson_coef is not None:
        print("Not accounting for Wilson...")

    obs_si = ['FL', 'AFB', 'S3', 'S4', 'S5', 'S7', 'S8', 'S9']
    wc_np = flavio.WilsonCoefficients()

    #This is the SM
    wc_np.set_initial({'C9_bsmumu' : 0., 'C10_bsmumu' : 0.}, scale = 100)

    si = {obs: flavio.np_prediction('%s(B0->K*mumu)' % obs, wc_np, vec['q2']) for obs in obs_si}

    return compute_J(si, vec['k'], vec['l'], vec['p'])


def compute_J(si, k, l, p):
    ''' Computes the differential branching factor 
    
    si : dict
        dictionary of all S_i terms
    k : float
        theta_k
    l : float
        theta l
    p : float
        phi
    '''

    fl =  3/4*(1 - si['FL']) * (np.sin(k) ** 2) + si['FL'] * (np.cos(k) ** 2) + 1/4 * (1 - si['FL']) * (np.sin(k) ** 2) * np.cos(2*l) - si['FL'] * (np.cos(k) ** 2) * np.cos(2*l)
    s3 = si['S3'] * (np.sin(k) ** 2) * (np.sin(l) ** 2) * np.cos(2 * p)
    s4 = si['S4'] * np.sin(2 * k) * np.sin(2*l) * np.cos(p)
    s5 = si['S5'] * np.sin(2 * k) * np.sin(l) * np.cos(p)
    afb = 4/3 * si['AFB'] * (np.sin(k) ** 2) * np.cos(l)
    s7 = si['S7'] * np.sin(2 * k) * np.sin(l) * np.sin(p)
    s8 = si['S8'] * np.sin(2 * k) * np.sin(2 * l) * np.sin(p)
    s9 = si['S9'] * (np.sin(k) ** 2) * (np.sin(l) ** 2) * np.sin(2 * p)

    return sum([fl, s3, s4, s5, afb, s7, s8, s9])

def format_range(x, a, b):
    ''' given uniform x in range [0,1], ouptut uniform in range [a,b] '''
    return x * (b - a) + a
# %%
J_min, J_max = 0,1.7
q2_min, q2_max = 1, 20
k_min, k_max = 0, 2 * np.pi
l_min, l_max = 0, 2 * np.pi
p_min, p_max = 0, 2 * np.pi

data_points = []

for i in tqdm(range(3000000)):
    # 1. generate random J
    J_rnd = np.random.random() * 1.7

    # 2. gnerate random kinematic vector
    data_vector = {
        'q2':np.random.uniform(q2_min, q2_max), 
        'k':np.random.uniform(k_min, k_max),
        'l':np.random.uniform(l_min, l_max),
        'p':np.random.uniform(p_min, p_max),
        } # verified to be uniform

    # 3. compute J from `data_vector`
    J_comp = compute_J_from_vec(data_vector)

    # 4. compare to random J
    if J_rnd < J_comp:
        data_vector['J_comp'] = J_comp
        data_points.append(data_vector)

# %%
data_points = pd.DataFrame(data_points)
# %%
f,ax=plt.subplots(2,2, figsize=(15,10))

# data_points.plot.hist(bins=50, density=True, ax=ax)

for i in range(2):
    for j in range(2):
        data_points[data_points.columns[2 * i + j]].plot.hist(bins=50, density=True, ax=ax[i,j])
        # if (j + i) != 0:
        #     ax[i,j].xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
        #     ax[i,j].xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
        #     ax[i,j].xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
# %%
jj = data_points.apply(lambda x: x, axis=0)
# %%
