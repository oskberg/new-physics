import numpy as np
import flavio
from tqdm import tqdm
import pandas as pd
import datetime


def compute_J_from_vec(vec, wilson_coef=None):
    if wilson_coef is not None:
        print("Not accounting for Wilson...")

    obs_si = ['FL', 'AFB', 'S3', 'S4', 'S5', 'S7', 'S8', 'S9']
    wc_np = flavio.WilsonCoefficients()

    # This is the SM
    wc_np.set_initial({'C9_bsmumu': 0., 'C10_bsmumu': 0.}, scale=100)

    si = {obs: flavio.np_prediction(
        '%s(B0->K*mumu)' % obs, wc_np, vec['q2']) for obs in obs_si}

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


def format_range(x, a, b):
    ''' given uniform x in range [0,1], ouptut uniform in range [a,b] '''
    return x * (b - a) + a


''''''
# DEFINE CONSTANTS

c9, c10 = 0, 0
J_min, J_max = 0, 1.8
# q2_min, q2_max = 1, 19
k_min, k_max = 0, np.pi
l_min, l_max = 0, np.pi
p_min, p_max = -np.pi, np.pi

data_points = []
excluded = 0
included = 0

wc_np = flavio.WilsonCoefficients()

c9_busmsm = float(input('C9_bsmumu = '))
c10_busmsm = float(input('C10_bsmumu = '))
q2_min = float(input('q2_min = '))
q2_max = float(input('q2_max = '))
# This is the SM
wc_np.set_initial(
    {'C9_bsmumu': c9_busmsm, 'C10_bsmumu': c10_busmsm}, scale=100)


for i in tqdm(range(int(input('datapoints = ')))):
    # 1. generate random J
    J_rnd = np.random.random() * 1.7

    # 2. generate random kinematic vector
    data_vector = {
        'q2': np.random.uniform(q2_min, q2_max),
        'k': np.random.uniform(k_min, k_max),
        'l': np.random.uniform(l_min, l_max),
        'p': np.random.uniform(p_min, p_max),
    }  # verified to be uniform

    dBR = flavio.np_prediction('dBR/dq2(B+->K*mumu)', wc_np, data_vector['q2'])

    dBR_rnd = np.random.uniform(0, 1e-7)

    if dBR_rnd < dBR:
        # 3. compute J from `data_vector`
        J_comp = compute_J_from_vec(data_vector)

        # 4. compare to random J
        if J_rnd < J_comp:
            data_vector['J_comp'] = J_comp
            data_points.append(data_vector)
            included += 1
        else:
            excluded += 1
    else:
        excluded += 1

data_points = pd.DataFrame(data_points)

date = datetime.datetime.now()

data_points.to_csv(
    f'/Users/oskar/MSci/new-physics/toy_data/flavio_tests/data/toy_data_c9_{c9_busmsm}_c10_{c10_busmsm}_{date.year}_{date.month}_{date.day}_{date.hour}.csv')

# pd.DataFrame({'a':[1,1],'b':[2,1]}).to_csv('data/blah.csv')
