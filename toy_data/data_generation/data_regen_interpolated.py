''' This script generates data using interpolation of the flavio.np_prediction for the observables. This should speed up the function significantly'''

import datetime

import flavio
import numpy as np
import pandas as pd
from tqdm import tqdm


def compute_br_interpolated():
    pass


# DEFINE CONSTANTS

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
wc_np.set_initial(
    {'C9_bsmumu': c9_busmsm, 'C10_bsmumu': c10_busmsm}, scale=100)

# perform 
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
        J_comp = compute_br_interpolated(data_vector)

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
