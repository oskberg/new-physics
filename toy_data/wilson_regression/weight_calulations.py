import flavio
import numpy as np
import pandas as pd

def compute_J_from_vec(vec, wilson_coef=None):
    obs_si = ['FL', 'AFB', 'S3', 'S4', 'S5', 'S7', 'S8', 'S9']
    
    if wilson_coef is None:
        print("Setting wilson to sm...")
        #This is the SM
        wilson_coef = flavio.WilsonCoefficients()
        wilson_coef.set_initial({'C9_bsmumu' : 0., 'C10_bsmumu' : 0.}, scale = 100)

    si = {obs: flavio.np_prediction('%s(B0->K*mumu)' % obs, wilson_coef, vec['q2']) for obs in obs_si}

    return compute_J(si, vec['k'], vec['l'], vec['p'])


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

    fl =  3/4*(1 - si['FL']) * (np.sin(k) ** 2) + si['FL'] * (np.cos(k) ** 2) + 1/4 * (1 - si['FL']) * (np.sin(k) ** 2) * np.cos(2*l) - si['FL'] * (np.cos(k) ** 2) * np.cos(2*l)
    s3 = si['S3'] * (np.sin(k) ** 2) * (np.sin(l) ** 2) * np.cos(2 * p)
    s4 = si['S4'] * np.sin(2 * k) * np.sin(2*l) * np.cos(p)
    s5 = si['S5'] * np.sin(2 * k) * np.sin(l) * np.cos(p)
    afb = 4/3 * si['AFB'] * (np.sin(k) ** 2) * np.cos(l)
    s7 = si['S7'] * np.sin(2 * k) * np.sin(l) * np.sin(p)
    s8 = si['S8'] * np.sin(2 * k) * np.sin(2 * l) * np.sin(p)
    s9 = si['S9'] * (np.sin(k) ** 2) * (np.sin(l) ** 2) * np.sin(2 * p)

    return sum([fl, s3, s4, s5, afb, s7, s8, s9])

def compute_br_array(observables, k, l, p):
    fl =  3/4*(1 - observables[0]) * (np.sin(k) ** 2) + observables[0] * (np.cos(k) ** 2) + 1/4 * (1 - observables[0]) * (np.sin(k) ** 2) * np.cos(2*l) - observables[0] * (np.cos(k) ** 2) * np.cos(2*l)
    s3 = observables[2] * (np.sin(k) ** 2) * (np.sin(l) ** 2) * np.cos(2 * p)
    s4 = observables[3] * np.sin(2 * k) * np.sin(2*l) * np.cos(p)
    s5 = observables[4] * np.sin(2 * k) * np.sin(l) * np.cos(p)
    afb = 4/3 * observables[1] * (np.sin(k) ** 2) * np.cos(l)
    s7 = observables[5] * np.sin(2 * k) * np.sin(l) * np.sin(p)
    s8 = observables[6] * np.sin(2 * k) * np.sin(2 * l) * np.sin(p)
    s9 = observables[7] * (np.sin(k) ** 2) * (np.sin(l) ** 2) * np.sin(2 * p)

    return sum([fl, s3, s4, s5, afb, s7, s8, s9])

def compute_J_from_df(df, w):
    obs_si = ['FL', 'AFB', 'S3', 'S4', 'S5', 'S7', 'S8', 'S9']
    
    # si = {obs: flavio.np_prediction('%s(B0->K*mumu)' % obs, w, df['q2']) for obs in obs_si}
    si = {}
    for o in obs_si:
        si[o] = df['q2'].apply(
            lambda q2: flavio.np_prediction(f'{o}(B0->K*mumu)', w, q2)
        )
    si = pd.DataFrame(si)
    return compute_J(si, df['k'], df['l'], df['p'])


def format_range(x, a, b):
    ''' given uniform x in range [0,1], ouptut uniform in range [a,b] '''
    return x * (b - a) + a