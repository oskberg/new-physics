#  %%
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPRegressor 
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
# %%

data_points0 = pd.read_csv('data/toy_data_c9_-0.46_c10_0.46_2021_10_29_13.csv', index_col=0)
data_points46 = pd.read_csv('data/toy_data_c9_0_c10_0_2021_10_29_11.csv', index_col=0)

data_points0['C9'] = [0] * data_points0.shape[0]
data_points46['C9'] = [-0.46] * data_points46.shape[0]
# data_points0['C10'] = [0] * data_points0.shape[0]
# data_points03['C10'] = [0.3] * data_points03.shape[0]

merged = pd.concat([data_points0, data_points46]).reset_index(drop=True)

# x = data_points.drop(columns=['J_comp'])
# y = data_points['J_comp']

# x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25)

merged
scaler = StandardScaler()

x = merged.drop(columns=['C9'])
y = merged[['C9']]

x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.20,shuffle=True)

y_train = y_train.replace({-0.46:0, 0:1, 0.46:2})

# model = DecisionTreeRegressor()
# model = DecisionTreeClassifier()
# model = LinearRegression()
model = MLPClassifier()
model.fit(x_train, y_train.values)
predictions = model.predict(x_test)
accuracy_score(predictions, y_test.replace({-0.46:0, 0:1, 0.46:2}))

# %%

# %%

# %%
