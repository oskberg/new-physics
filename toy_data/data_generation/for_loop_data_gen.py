''' This script generates data using interpolation of the flavio.np_prediction for the observables. This should speed up the function significantly'''

import datetime
from random import random
from xml.dom.minidom import ReadOnlySequentialNamedNodeMap

import flavio
import numpy as np
import pandas as pd
from tqdm import tqdm

import pickle
import scipy.interpolate as interp


# def compute_J_from_vec(vec, wilson_coef=None):
#     if wilson_coef is not None:
#         print("Not accounting for Wilson...")

#     obs_si = ['FL', 'AFB', 'S3', 'S4', 'S5', 'S7', 'S8', 'S9']
#     wc_np = flavio.WilsonCoefficients()

#     # This is the SM
#     wc_np.set_initial({'C9_bsmumu': 0., 'C10_bsmumu': 0.}, scale=100)

#     si = {obs: flavio.np_prediction(
#         '%s(B0->K*mumu)' % obs, wc_np, vec['q2']) for obs in obs_si}

#     return compute_J(si, vec['k'], vec['l'], vec['p'])

def compute_br(si, k, l, p):
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

    fl = 3/4*(1 - si['FL']) * (np.sin(k) ** 2) + si['FL'] * (np.cos(k) ** 2) + 1/4 * (1 -
                                                                                      si['FL']) * (np.sin(k) ** 2) * np.cos(2*l) - si['FL'] * (np.cos(k) ** 2) * np.cos(2*l)
    s3 = si['S3'] * (np.sin(k) ** 2) * (np.sin(l) ** 2) * np.cos(2 * p)
    s4 = si['S4'] * np.sin(2 * k) * np.sin(2*l) * np.cos(p)
    s5 = si['S5'] * np.sin(2 * k) * np.sin(l) * np.cos(p)
    afb = 4/3 * si['AFB'] * (np.sin(k) ** 2) * np.cos(l)
    s7 = si['S7'] * np.sin(2 * k) * np.sin(l) * np.sin(p)
    s8 = si['S8'] * np.sin(2 * k) * np.sin(2 * l) * np.sin(p)
    s9 = si['S9'] * (np.sin(k) ** 2) * (np.sin(l) ** 2) * np.sin(2 * p)

    return sum([fl, s3, s4, s5, afb, s7, s8, s9])


def compute_br_interpolated(df_og, obs_dict):

    df = df_og.copy()

    q_in = obs_dict['q_range']
    c9_in = obs_dict['c9_range']
    c10_in = obs_dict['c10_range']

    Q_in = obs_dict['q_grid']
    C9_in = obs_dict['c9_grid']
    C10_in = obs_dict['c10_grid']

    eval_points_q, eval_points_c9, eval_points_c10 = df['q2'], df['c9'], df['c10']

    for ob in obs_si:
        true_obs_values = obs_dict[ob]

        interpolated_obs_values = interp.interpn(
            [q_in, c9_in, c10_in],
            true_obs_values,
            (eval_points_q, eval_points_c9, eval_points_c10),
            method='linear'
        )

        df[ob] = interpolated_obs_values

    df['BR'] = compute_br(df, df['k'], df['l'], df['p'])

    return df['BR']

# DEFINE CONSTANTS
obs_si = ['FL', 'AFB', 'S3', 'S4', 'S5', 'S7', 'S8', 'S9']

J_min, J_max = 0, 1.8
# q2_min, q2_max = 1, 19
k_min, k_max = 0, np.pi
l_min, l_max = 0, np.pi
p_min, p_max = -np.pi, np.pi

data_points = []
excluded = 0
included = 0

wc_np = flavio.WilsonCoefficients()


q2_min = float(input('q2_min = '))
q2_max = float(input('q2_max = '))

# read in interpolation values
observable_data_path = '/Users/jakubpazio/Imperial/Master Project/new-physics/new-physics/toy_data/data_generation/interpolation/interp_2022_1_27_13'
# observable_data_path = 'data/interpolation/interp_2022_1_27_13'
with open(observable_data_path, 'rb') as infile:
    observable_dict_import = pickle.load(infile)

min_br, max_br = 0, 1.7
min_dbr, max_dbr = 0, 1e-7

number_data_points = int(input('datapoints = '))

# GENERATE THE DATA


# for c9_t in np.linspace(-1,1,10):
#     for c10_t in np.linspace(-1,1,10):

c9_t = float(input('c9 = '))
for i in range(2):
    c10_t = float(input('c10 = '))

    print(c9_t,c10_t)

    wc_np.set_initial({'C9_bsmumu': c9_t, 'C10_bsmumu': c10_t}, scale=100)

    print('check1')   
    random_data = pd.DataFrame({
        'q2': np.random.uniform(q2_min, q2_max, number_data_points),
        'k': np.random.uniform(k_min, k_max, number_data_points),
        'l': np.random.uniform(l_min, l_max, number_data_points),
        'p': np.random.uniform(p_min, p_max, number_data_points),
        'c9': c9_t * number_data_points,
        'c10': c10_t * number_data_points,
    })

    print('check2')   


    random_data['BR_rnd'] = np.random.uniform(min_br, max_br, number_data_points)
    random_data['dBR_rnd'] = np.random.uniform(min_dbr, max_dbr, number_data_points)
    random_data['dBR'] = random_data['q2'].apply(lambda q2: flavio.np_prediction('dBR/dq2(B+->K*mumu)', wc_np, q2))
    print('check3') 
    q2_filtered_data = random_data[random_data['dBR_rnd'] < random_data['dBR']].copy()
    print('check3.5') 
    q2_filtered_data['BR_interpolated'] = compute_br_interpolated(q2_filtered_data[['q2', 'k', 'l', 'p', 'c9', 'c10']], observable_dict_import)
    print('check4') 
    completely_filtered_data = q2_filtered_data[q2_filtered_data['BR_rnd'] < q2_filtered_data['BR_interpolated']]


    date = datetime.datetime.now()

    completely_filtered_data.to_csv(
        f'/Users/jakubpazio/Imperial/Master Project/new-physics/new-physics/toy_data/data_generation/datasets/toy_data_c9_{c9_t}_c10_{c10_t}_{date.year}_{date.month}_{date.day}_{date.hour}.csv', index=False)

