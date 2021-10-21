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
# %%
