# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # krigingを実装する

import calc_variogram as variogram
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import scipy.optimize as opt
import matplotlib.pyplot as plt

# データの用意
data = pd.read_csv('/Users/LOng/sengokulab/LDAT_LS_2019/data/dev/house_price_raw.csv')
data['price'] = np.log(data.price)
data = data[['px', 'py', 'price']].values

vario = variogram.get_diff(data)

plt.scatter(vario[0], vario[1], facecolors = 'None', edgecolors = 'blue')

param, lag_num, fig = variogram.auto_vario(vario, range(5,13))

fig

print(param)
print(lag_num)

from scipy.interpolate import Rbf

x = data[:, 0:1]
y = data[:, 1:2]
z = data[:, 2:3]

rbfi = Rbf(x, y, z)


