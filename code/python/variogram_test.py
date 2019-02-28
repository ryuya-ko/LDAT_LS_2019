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

# # scipyを用いてvariogram, krigingを実装する

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy import stats
import matplotlib.pyplot as plt

# ## scipyを用いた距離行列の作成・バリオグラムのプロット

n = 3
data = np.array([[np.random.normal(0, 10) for i in range(n)] for i in range(100)]) #x,y,zを生成

print(data)

data[:,0:2]

dist_vec=pdist(data[:,0:2], 'euclidean')

print(dist_vec)

dist_mat = squareform(dist_vec)

print(dist_mat)

z_mat = squareform(pdist(data[:,2:], 'euclidean')**2/2)

[dist_mat, z_mat]

z_mat.shape

z_vec = squareform(z_mat)

print(z_vec)

np.stack([dist_vec, z_vec])

plt.scatter(dist_vec, pdist(data[:,2:], 'euclidean')**2/2, facecolors = 'None', edgecolors='blue')


def get_diff(data):
    '''
    get the difference of spatial data
    input: (n,3) matrix data
    output: np.array(n,2)
    '''
    dist_vec = pdist(data[:, :2])
    z_vec = pdist(data[:, 2:])**2/2
    diff = np.stack([dist_vec, z_vec])
    
    return diff


vario = get_diff(data)

plt.scatter(vario[0], vario[1],  facecolors = 'None', edgecolors='blue')


data = np.array([[np.random.normal(0, 10) for i in range(n)] for i in range(5000)])

# %timeit variogram(data)

# %timeit get_diff(data)

print(vario[0].flatten())

np.max(vario[0])

vario.shape


def emp_variogram(z_vario, lag_h):
    '''
    calculate empirical variogram
    input: difference of spatial (2, nC2)matrix,  bandwith of bins
    '''
    num_rank = int(np.max(z_vario[0]) / lag_h)
    bin_means, bin_edges, bin_number = stats.binned_statistic(z_vario[0], z_vario[1], statistic='mean', bins=num_rank)
    e_vario = np.stack([bin_edges[1:], bin_means[0:]], axis=0) #bin_edgesに関しては最初のものを省く
    e_vario = np.delete(e_vario, np.where(e_vario[1] <= 0), axis=1)
    
    return e_vario


print(emp_variogram(vario, 10))

e_vario = emp_variogram(vario,10)

e_vario[1]

plt.scatter(e_vario[0], e_vario[1])


def liner_model(x, a, b):
    return a + b * x
def gaussian_model(x, a, b, c):
    return a + b * (1 - np.exp(-(x / c)**2))
def exponential_model(x, a, b, c):
    return a + b * (1 - np.exp(-(x / c)))
def spherical_model(x, a, b, c):
    cond = [x < c, x > c]
    func = [lambda x : a + (b / 2)  * (3 * (x / c) - (x / c)**3), lambda x : a + b]
    return np.piecewise(x, cond, func)


import scipy.optimize as opt
def auto_fit(e_vario, fitting_range, selected_model):
    # フィッティングレンジまでで標本バリオグラムを削る
    data = np.delete(e_vario, np.where(e_vario[0]>fitting_range)[0], axis=1)
    if (selected_model == 0):
        param, cov = opt.curve_fit(liner_model, data[0], data[1])
    elif (selected_model == 1):
        param, cov = opt.curve_fit(gaussian_model, data[0], data[1], [0, 0, fitting_range])
    elif (selected_model == 2):
        param, cov = opt.curve_fit(exponential_model, data[0], data[1], [0, 0, fitting_range])
    elif (selected_model == 3):
        param, cov = opt.curve_fit(spherical_model, data[0], data[1], [0, 0, fitting_range])
    param = np.insert(param, 0, [selected_model,fitting_range])
    return param


param  = auto_fit(e_vario, 120, 0)

print(param)

fig = plt.figure()
ax = plt.subplot(111)
ax.plot(e_vario[0], e_vario[1], 'o')
xlim_arr = np.arange(0, np.max(e_vario[0]), 10)
if (param[0] == 0):
    ax.plot(xlim_arr, liner_model(xlim_arr, param[2], param[3]), 'r-')
    print(param[2], param[3])
elif (param[0] == 1):
    ax.plot(xlim_arr, gaussian_model(xlim_arr, param[2], param[3], param[4]), 'r-')
    print(xlim_arr, param[3], param[4])
elif (param[0] == 2):
    ax.plot(xlim_arr, exponential_model(xlim_arr, param[2], param[3], param[4]), 'r-')
    print(param[2], param[3], param[4])
elif (param[0] == 3):
    ax.plot(xlim_arr, spherical_model(xlim_arr, param[2], param[3], param[4]), 'r-')
    print(param[2], param[3], param[4])
# グラフのタイトルの設定
ax.set_title('Semivariogram')
# 軸ラベルの設定
#ax.set_xlim([0, np.max(e_vario[0])])
#ax.set_ylim([0, np.max(e_vario[1])])
ax.set_xlabel('Distance [m]')
ax.set_ylabel('Semivariance')
# グラフの縦横比を調整
aspect = 0.8 * (ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])                     
ax.set_aspect(aspect)
# グラフの描画
plt.show()
