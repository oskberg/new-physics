#%%
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
import flavio
import flavio.plots

#Supress the warnings about QCDF corrections above 6 being unreliable
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm

#%%
def print_observables(observables):
    for ob, val in observables.items():
        print(ob, val['val'], '±', val['err'])

#%%
def compute_I(si, k, l, p):
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

    fl =  3/4*(1 - si['FL']) * (np.sin(k) ** 2) + si['FL'] * (np.cos(k) ** 2) + 1/4 * (1 - si['FL']) * (np.sin(k) ** 2) * np.cos(2*l) - si['FL'] * (np.cos(k) ** 2) * np.cos(l)
    s3 = si['S3'] * (np.sin(k) ** 2) * (np.sin(l) ** 2) * np.cos(2 * p)
    s4 = si['S4'] * np.sin(2 * k) * np.sin(2*l) * np.cos(p)
    s5 = si['S5'] * np.sin(2 * k) * np.sin(l) * np.cos(p)
    afb = 4/3 * si['AFB'] * (np.sin(k) ** 2) * np.cos(l)
    s7 = si['S7'] * np.sin(2 * k) * np.sin(l) * np.sin(p)
    s8 = si['S8'] * np.sin(2 * k) * np.sin(2 * l) * np.sin(p)
    s9 = si['S9'] * (np.sin(k) ** 2) * (np.sin(l) ** 2) * np.sin(2 * p)

    return sum([fl, s3, s4, s5, afb, s7, s8, s9])

#%%
# wc_np = flavio.WilsonCoefficients()
# #This is the SM
# wc_np.set_initial({'C9_bsmumu' : 0., 'C10_bsmumu' : 0.}, scale = 160)

# obs_si = ['FL', 'AFB', 'S3', 'S4', 'S5', 'S7', 'S8', 'S9']
# stdPreds_si = {}
# q2Val = 6.0
# for _obs in obs_si:
#     stdPreds_si[_obs] = {}
#     stdPreds_si[_obs]['val'] = flavio.np_prediction('%s(B0->K*mumu)' % _obs, wc_np, q2Val)
#     stdPreds_si[_obs]['err'] = flavio.np_uncertainty('%s(B0->K*mumu)' % _obs, wc_np, q2Val)
# print_observables(stdPreds_si)
# %%
''' Perform grid search over the different angles '''
q2_min, q2_max = 1, 20
k_min, k_max = 0, 2 * np.pi
l_min, l_max = 0, 2 * np.pi
p_min, p_max = 0, 2 * np.pi

angle_steps = 20
q_steps = 20

obs_si = ['FL', 'AFB', 'S3', 'S4', 'S5', 'S7', 'S8', 'S9']
wc_np = flavio.WilsonCoefficients()

#This is the SM
wc_np.set_initial({'C9_bsmumu' : 0., 'C10_bsmumu' : 0.}, scale = 160)

# for each q2, generate the S_i variables

fac_values = []

for q2 in tqdm(np.linspace(q2_min, q2_max, q_steps)):
    si = {obs: flavio.np_prediction('%s(B0->K*mumu)' % obs, wc_np, q2) for obs in obs_si}

    for k in np.linspace(k_min, k_max, angle_steps):
        for l in np.linspace(l_min, l_max, angle_steps):
            for p in np.linspace(p_min, p_max, angle_steps):
                fac = compute_I(si, k, l, p)
                fac_values.append({'I':fac, 'q2':q2,'k':k, 'l':l, 'p':p})
                # fac_values.append({fac:[q2, k, l, p]})
# %%
df = pd.DataFrame(fac_values)

# %%
fig, axes = plt.subplots(2,2, figsize=(15,10))

counter = 0
labels = df.columns[1:]
for a in axes:
    for ax in a:
        v = labels[counter]
        uniques = df[v].unique()
        l = [df[df[v] == var]['I'].values for var in uniques]
        
        ax.violinplot(dataset = l,showmeans=False, showmedians=False,showextrema=False)
        ax.set_xlabel(v)
        ax.set_ylabel('I')

        counter += 1
# %%
# df.plot.scatter(x='q2', y='I', alpha=0.005, s=10)
# %%
df['I'].plot.hist(bins=200)
# %%
df[df['I']<0][df.columns[1:]].hist(alpha=1, bins=20, figsize=(10,7))
# %%
I_max = df.I.max()
# %%
I_max
# %%
