# %%
import pickle
from functools import lru_cache

import flavio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import weight_calulations as wc

# %%
weightd_sm = pd.read_csv('data/low_q_with_weights.csv')
clean_sm = weightd_sm[['q2', 'k', 'l', 'p', 'BR_sm']].copy()
rounded_sm = clean_sm.copy()
rounded_sm['q2'] = rounded_sm['q2'].round(2)
# %%
# variable_range = {
#     'q2':(0.5,2),
#     'l':(0,np.pi),
#     'k':(0,np.pi),
#     'p':(-np.pi,np.pi),
# }
# %%


@lru_cache(maxsize=1000)
def get_observable(key, q2, c9, c10):
    wilson_coef.set_initial({'C9_bsmumu': c9, 'C10_bsmumu': c10}, scale=100)

    observable = flavio.np_prediction(
        f'{key}(B0->K*mumu)', wilson_coef, q2
    )

    return observable


# %%
c_min, c_max = -1, 1
n_samples = 1000
sample_size = 1000
n_bins = 10
bin_edges_ql = [
    np.linspace(0.5, 2, n_bins+1),           # q2
    np.linspace(0, np.pi, n_bins+1),         # l
]
bin_edges_kl = [
    np.linspace(0, np.pi, n_bins+1),         # k
    np.linspace(0, np.pi, n_bins+1),         # l
]
bin_edges_qkl = [
    np.linspace(0.5, 2, n_bins+1),           # q2
    np.linspace(0, np.pi, n_bins+1),         # k
    np.linspace(0, np.pi, n_bins+1),         # l
]

obs_si = ['FL', 'AFB', 'S3', 'S4', 'S5', 'S7', 'S8', 'S9']

variables = ['q2', 'l']
# each node is  a bin in a grid of dimensions len(variables)
features_2d = [f'x{i}' for i in range(n_bins ** len(variables))]
targets = ['c9', 'c10']

dataset = []

wilson_coefficients = np.random.uniform(
    c_min, c_max, (n_samples, len(targets))
)


wilson_coef = flavio.WilsonCoefficients()

#%%
for c9, c10 in tqdm(wilson_coefficients):
    sample = rounded_sm.sample(sample_size)
    # set wilson coefficients
    sample['c9'] = [c9] * sample.shape[0]
    sample['c10'] = [c10] * sample.shape[0]

    # reweight the sample according to the wilson coefficients
    for obs in obs_si:
        sample[obs] = sample[['q2', 'c9', 'c10']].apply(
            lambda row: get_observable(obs, *row),
            axis=1,
            raw=True,
        )

    observables = sample[obs_si]
    k_df = sample['k']
    l_df = sample['l']
    p_df = sample['p']
    sample['BR_new'] = wc.compute_br(observables, k_df, l_df, p_df)

    sample['weight'] = sample['BR_new'] / sample['BR_sm']

    values_2d_ql, _ = np.histogramdd(
        sample[variables].values,
        bins=bin_edges_ql,
        weights=sample['weight'],
        density=True
    )

    values_2d_kl, _ = np.histogramdd(
        sample[['k', 'l']].values,
        bins=bin_edges_kl,
        weights=sample['weight'],
        density=True
    )

    values_3d_qkl, _ = np.histogramdd(
        sample[['q2', 'k', 'l']].values,
        bins=bin_edges_qkl,
        weights=sample['weight'],
        density=True
    )

    dataset.append({
        'histogram_qkl': values_3d_qkl,
        'histogram_ql': values_2d_ql,
        'histogram_kl': values_2d_kl,
        'c9': c9,
        'c10': c10
    })

# %%
dataset
# %%
output_file_name = 'low_q_binned_weights_1000_new_samples.pkl'
path_to_file = 'data/'
with open(path_to_file + output_file_name, 'wb') as out_file:
    pickle.dump(dataset, out_file)

# %%
points_per_axis = 10
number_of_features = 3
grid = np.ones([points_per_axis] * number_of_features)

q_range = np.linspace(0.5,2, points_per_axis)
l_range = np.linspace(0,np.pi, points_per_axis)
k_range = np.linspace(0,np.pi, points_per_axis)
p_range = np.linspace(-np.pi,np.pi, points_per_axis)

x, y, z = np.meshgrid(q_range, k_range, l_range)
x_2d, y_2d = np.meshgrid(k_range, l_range)
# %%
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

iii = 9

sm_grid = dataset[iii]['histogram_qkl']
print(dataset[iii]['c9'])
print(dataset[iii]['c10'])

# sm_grid = og_values

ax.scatter(x, y, z, c = sm_grid, s=np.where(sm_grid > 0.5e-1, 10, 0), alpha=0.8)
ax.set_xlabel('q2')
ax.set_ylabel('k')
ax.set_zlabel('l')
# %%
og_values, _ = np.histogramdd(
        clean_sm[['q2', 'k', 'l']].values,
        bins=bin_edges_qkl,
        density=True
    )
# %%
