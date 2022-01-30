#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import flavio
import weight_calulations as wc

import swifter
# %%
weightd_sm = pd.read_csv('data/low_q_with_weights.csv')
clean_sm = weightd_sm[['q2', 'k','l','p','BR_sm']].copy()
# %%
# sm_with_c = clean_sm.iloc[:5000]
# c9 = 1
# c10 = -1
# df_len = sm_with_c.shape[0]
# sm_with_c['c9'] = [0] * int(df_len/2) + [c9] * int(df_len/2)
# sm_with_c['c10'] = [0] * int(df_len/2) + [c10] * int(df_len/2)
# %%

wilson_coefficients = [
    {'c9':0,'c10':0},
    {'c9':1,'c10':-1},
    {'c9':2,'c10':-2},
    {'c9':3,'c10':-3},
    {'c9':4,'c10':-4},
    {'c9':5,'c10':-5},
    {'c9':6,'c10':-6},
    {'c9':7,'c10':-7},
    {'c9':8,'c10':-8},
    {'c9':9,'c10':-9},
    {'c9':10,'c10':-10},
    # {'c9':100,'c10':-100},
]
# %%
sub_sample_len = 60
datasets = []
for wilson_set in wilson_coefficients:
    cut_down_dataset = clean_sm.iloc[:sub_sample_len].copy()
    cut_down_dataset['c9'] = [wilson_set['c9']] * sub_sample_len
    cut_down_dataset['c10'] = [wilson_set['c10']] * sub_sample_len
    datasets.append(cut_down_dataset)
    print(wilson_set)
total_dataset = pd.concat(datasets, axis=0).reset_index(drop=True)
# %%
wilson_coef = flavio.WilsonCoefficients()
def get_observable(key, q2, c9, c10):
    wilson_coef.set_initial({'C9_bsmumu' : c9, 'C10_bsmumu' : c10}, scale = 100)

    observable = flavio.np_prediction(
        f'{key}(B0->K*mumu)', wilson_coef, q2
        )
    return observable

# sm_short = sm_with_c.iloc[:1000].copy()

obs_si = ['FL', 'AFB', 'S3', 'S4', 'S5', 'S7', 'S8', 'S9']

for obs in obs_si[:]:
    total_dataset[obs] = total_dataset[['q2', 'c9', 'c10']].apply(
        lambda row: get_observable(obs, *row),
        axis=1,
        raw=True
    )
# %%
observables = total_dataset[obs_si]
k_df = total_dataset['k']
l_df = total_dataset['l']
p_df = total_dataset['p']
total_dataset['BR_new'] = wc.compute_br(observables, k_df, l_df, p_df)
# %%
total_dataset['weight'] = total_dataset['BR_new'] / total_dataset['BR_sm']
# %%
total_dataset
# %%
plt.hist(total_dataset.loc[total_dataset['c9'] == 0, 'l'], weights=total_dataset.loc[total_dataset['c9'] == 0, 'weight'], density=True, alpha=0.6, bins=40)
plt.hist(total_dataset.loc[total_dataset['c9'] != 0, 'l'], weights=total_dataset.loc[total_dataset['c9'] != 0, 'weight'], density=True, alpha=0.6, bins=40)
# %%
total_dataset.to_csv('low_q_swifter_uniform_weight.csv')
# %%
