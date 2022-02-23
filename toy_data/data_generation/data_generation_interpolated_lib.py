import datetime
import pickle

import numpy as np
import pandas as pd
import scipy.interpolate as interp


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


class InterpolatedDataGenerator():
    def __init__(self, data_path) -> None:
        # DEFINE CONSTANTS
        self.obs_si = ['FL', 'AFB', 'S3', 'S4', 'S5', 'S7', 'S8', 'S9']

        self.BR_min, self.BR_max = 0, 1.8
        self.min_br, self.max_br = 0, 1.7
        self.min_dbr, self.max_dbr = 0, 1e-7

        self.q2_min, self.q2_max = 0.5, 6
        self.k_min, self.k_max = 0, np.pi
        self.l_min, self.l_max = 0, np.pi
        self.p_min, self.p_max = -np.pi, np.pi

        self.load_interp_data(data_path)


    def load_interp_data(self, data_path):
        with open(data_path, 'rb') as data_file:
            self.observable_dict = pickle.load(data_file)

        self.q_in = self.observable_dict['q_range']
        self.c9_in = self.observable_dict['c9_range']
        self.c10_in = self.observable_dict['c10_range']

        self.true_br_vals = self.observable_dict['dBR/dq2']


    def interpolate_dBR(self):
        return interp.interpn(
            [self.q_in, self.c9_in, self.c10_in],
            self.true_br_vals,
            (self.eval_points_q, self.eval_points_c9, self.eval_points_c10),
            method='linear'
        )


    def interpolate_BR(self):
        interpolated_observables = {}
        for ob in self.obs_si:
            true_obs_values = self.observable_dict[ob]

            interpolated_obs_values = interp.interpn(
                [self.q_in, self.c9_in, self.c10_in],
                true_obs_values,
                (self.eval_points_q, self.eval_points_c9, self.eval_points_c10),
                method='linear'
            )

            interpolated_observables[ob] = interpolated_obs_values

        return compute_br(interpolated_observables, self.q2_filtered_data['k'], self.q2_filtered_data['l'], self.q2_filtered_data['p'])


    def generate_data(self, c9, c10, initial_points, clean=False):
        self.c9, self.c10 = c9, c10
        self.random_data = pd.DataFrame({
            'q2': np.random.uniform(self.q2_min, self.q2_max, initial_points),
            'k': np.random.uniform(self.k_min, self.k_max, initial_points),
            'l': np.random.uniform(self.l_min, self.l_max, initial_points),
            'p': np.random.uniform(self.p_min, self.p_max, initial_points),
            'c9': [self.c9] * initial_points,
            'c10': [self.c10] * initial_points,
        })

        self.random_data['BR_rnd'] = np.random.uniform(self.min_br, self.max_br, initial_points)
        self.random_data['dBR_rnd'] = np.random.uniform(self.min_dbr, self.max_dbr, initial_points)

        self.eval_points_q = self.random_data['q2']
        self.eval_points_c9 = self.random_data['c9']
        self.eval_points_c10 = self.random_data['c10']

        self.random_data['dBR'] = self.interpolate_dBR()

        self.q2_filtered_data = self.random_data[self.random_data['dBR_rnd'] < self.random_data['dBR']].copy()

        self.eval_points_q = self.q2_filtered_data['q2']
        self.eval_points_c9 = self.q2_filtered_data['c9']
        self.eval_points_c10 = self.q2_filtered_data['c10']

        self.q2_filtered_data['BR_interpolated'] = self.interpolate_BR()

        self.completely_filtered_data = self.q2_filtered_data[self.q2_filtered_data['BR_rnd'] < self.q2_filtered_data['BR_interpolated']]

        if clean:
            return self.completely_filtered_data.drop(columns=['BR_rnd', 'dBR_rnd', 'dBR', 'BR_interpolated'])
        else:
            return self.completely_filtered_data

    
    def to_csv(self, directory_path):
        date = datetime.datetime.now()

        self.completely_filtered_data.to_csv(directory_path + 
            f'/c9_{self.c9}_c10_{self.c10}_{date.year}_{date.month}_{date.day}_{date.hour}.csv', index=False)
